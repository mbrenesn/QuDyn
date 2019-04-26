// Quantum Trajectories

#include <iostream>
#include <sstream>

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"
#include "../Operators/AdjacencyOp.h"
#include "../Operators/DiagonalOp.h"
#include "../InitialState/InitialState.h"
#include "../TimeEvo/KrylovEvo.h"
#include "../TimeEvo/ChebyshevEvo.h"

#include <petsctime.h>

std::vector<double> linspace(double a, double b, size_t n) {
    double h = (b - a) / static_cast<double>(n - 1);
    std::vector<double> vec(n);
    typename std::vector<double>::iterator x;
    double val;
    for (x = vec.begin(), val = a; x != vec.end(); ++x, val += h)
        *x = val;
    return vec;
}

int main(int argc, char **argv)
{
  unsigned int l = 777;
  unsigned int n = 777;
  double alpha = 0.777;
  double delta = 0.777;
  double h = 0.777;
  double dephase_gamma = 0.777;
  double max_time = 0.777;
  unsigned int time_its = 777;

  if(argc < 8){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --dephase_gama [d] --max_time [tmax] --time_its [its] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--l") l = atoi(argv[i + 1]);
    else if(str == "--n") n = atoi(argv[i + 1]);
    else if(str == "--alpha") alpha = atof(argv[i + 1]);
    else if(str == "--delta") delta = atof(argv[i + 1]);
    else if(str == "--h") h = atof(argv[i + 1]);
    else if(str == "--dephase_gamma") dephase_gamma = atof(argv[i + 1]);
    else if(str == "--max_time") max_time = atof(argv[i + 1]);
    else if(str == "--time_its") time_its = atoi(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || alpha == 0.777 || delta == 0.777 || h == 0.777 || dephase_gamma == 0.777 || 
          max_time == 0.777 || time_its == 777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --dephase_gamma [d] --max_time [tmax] --time_its [its] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  std::vector<double> h_vec(l, h);
  
  // Establish the environment
  Environment env(argc, argv, l, n);

  PetscMPIInt mpirank = env.mpirank;
  PetscMPIInt mpisize = env.mpisize;

  // Establish the basis environment, by pointer, to call an early destructor and reclaim
  // basis memory
  Basis *basis = new Basis(env);

  // Construct basis
  basis->construct_int_basis();
  //basis->print_basis(env);

  AdjacencyOp adjmat(env, *basis, alpha, true); 

  DiagonalOp diagop(env, *basis, true, false);
  diagop.construct_xxz_diagonal(basis->int_basis, delta, h_vec);

  // Diagonal part of the effective Hamiltonian
  // Dephasing part in Vec dephase
  Vec dephase;
  VecDuplicate(diagop.DiagonalVec, &dephase);
  for(PetscInt ii = basis->start; ii < basis->end; ++ii){
    PetscScalar val = PETSC_i * dephase_gamma * 0.5;
    VecSetValue(dephase, ii, val, INSERT_VALUES);
  }

  // Add the Hamiltonian with the dephasing part
  PetscScalar alp = -1.0;
  VecAXPY(diagop.DiagonalVec, alp, dephase);

  VecDestroy(&dephase);

  // Now put the effective Hamiltonian in AdjacencyMat
  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  // Initial state, Neel state
  Vec initial;
  VecDuplicate(diagop.DiagonalVec, &initial);
  VecZeroEntries(initial);
  LLInt index = Utils::get_neel_index(env, *basis);
  VecSetValue(initial, index, 1.0, INSERT_VALUES);
  VecAssemblyBegin(initial);
  VecAssemblyEnd(initial);
  
  delete basis;

  // Time Evo
  double tol = 1.0e-7;
  int maxits = 1000000;
  int iterations = time_its + 1;
  KrylovEvo te(adjmat.AdjacencyMat, tol, maxits);

  std::vector<double> times = linspace(0.0, max_time, iterations);
  
  // Initial values
  PetscScalar imbalance;
  PetscReal norm;
  Vec init_c;
  VecDuplicate(initial, &init_c);
  VecPointwiseMult(init_c, diagop.TotalZ, initial);
  VecDot(init_c, initial, &imbalance);

  std::cout << std::fixed;
  std::cout.precision(7);
  if(mpirank == 0){
    std::cout << PetscRealPart(imbalance) << std::endl;
  }

  // Trajectories procedure
  boost::random::mt19937 gen;
  gen.seed(static_cast<LLInt>(std::time(0)));
  boost::random::uniform_real_distribution<double> r_dist(0, 1);
  boost::random::uniform_int_distribution<int> i_dist(0, l - 1);

//  for(unsigned int tt = 1; tt < (iterations); ++tt){
    // A copy of the initial state
    VecCopy(initial, init_c);
  
    // First step: Compute evolution under effective Hamiltonian
    //te.krylov_evo(times[tt], times[tt - 1], initial);
    te.krylov_evo(0.1, 0.0, initial);

    // Compute the norm
    VecNorm(initial, NORM_2, &norm);

    // < phi | phi > = 1 - p
    double p = 1.0 - norm; 

    // double r1 = r_dist(gen);
    double r1; int m;
    if(mpirank == 0){
      r1 = r_dist(gen);
      m = i_dist(gen);
    }
    MPI_Bcast(&r1, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    // No jump
    if(r1 > p){
      // Normalise state
      VecNormalize(initial, &norm);
    }
    // Jump
    else{
      // Pick a jump operator and apply it to initial state
      VecPointwiseMult(initial, diagop.SigmaZ[m], init_c);
    }

    // Evaluate expectation value, imbalance in this case
    VecPointwiseMult(init_c, diagop.TotalZ, initial);
    VecDot(init_c, initial, &imbalance);

    if(mpirank == 0)
      std::cout << PetscRealPart(imbalance) << std::endl;
//  }
  
  VecDestroy(&initial);
  VecDestroy(&init_c);
  return 0;
}
