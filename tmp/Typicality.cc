// Typicallity

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

#include <boost/random/normal_distribution.hpp> 

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
  double max_time = 0.777;
  unsigned int time_its = 777;
  std::cout << std::fixed;
  std::cout.precision(7);

  if(argc < 7){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --max_time [tmax] --time_its [its] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--l") l = atoi(argv[i + 1]);
    else if(str == "--n") n = atoi(argv[i + 1]);
    else if(str == "--alpha") alpha = atof(argv[i + 1]);
    else if(str == "--delta") delta = atof(argv[i + 1]);
    else if(str == "--h") h = atof(argv[i + 1]);
    else if(str == "--max_time") max_time = atof(argv[i + 1]);
    else if(str == "--time_its") time_its = atoi(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || alpha == 0.777 || delta == 0.777 || h == 0.777 || 
          max_time == 0.777 || time_its == 777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --max_time [tmax] --time_its [its] -[PETSc/SLEPc options]" << std::endl;
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

  DiagonalOp diagop(env, *basis, false, false);
  diagop.construct_xxz_diagonal(basis->int_basis, delta, h_vec);

  // Now put the effective Hamiltonian in AdjacencyMat
  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  // Initial state, random
  Vec initial;
  VecDuplicate(diagop.DiagonalVec, &initial);

  boost::random::mt19937 gen;
  gen.seed(mpirank);
  int rtime = 1;
  if(rtime) gen.seed(static_cast<LLInt>(std::time(0) + mpirank));
  boost::random::normal_distribution<double> dist(0.0, 1.0);
  for(PetscInt index = basis->start; index < basis->end; ++index){
    double a = dist(gen);
    double b = dist(gen);
    PetscScalar c = a + (PETSC_i * b);
    VecSetValue(initial, index, c, INSERT_VALUES);
  }
  VecAssemblyBegin(initial);
  VecAssemblyEnd(initial);
  // Normalisation
  PetscReal norm;
  VecNorm(initial, NORM_2, &norm);
  VecNormalize(initial, &norm);
  //VecView(initial, PETSC_VIEWER_STDOUT_WORLD);

  delete basis;

  //MatView(adjmat.AdjacencyMat, PETSC_VIEWER_STDOUT_WORLD);
  //MatView(adjmat.CurrentMat, PETSC_VIEWER_STDOUT_WORLD);

  // Time Evo
  double tol = 1.0e-7;
  int maxits = 1000000;
  int iterations = time_its + 1;
  KrylovEvo te(adjmat.AdjacencyMat, tol, maxits);

  std::vector<double> times = linspace(0.0, max_time, iterations);

  // Typicallity procedure
  // Initial values
  PetscScalar c_s;
  Vec phi_1, phi_2, vec_help;
  VecDuplicate(initial, &phi_1);
  VecDuplicate(initial, &phi_2);
  VecDuplicate(initial, &vec_help);
   
  // Phi1 and Phi2
  VecCopy(initial, phi_1);
  MatMult(adjmat.CurrentMat, initial, phi_2);

  // Initial expectation value
  MatMult(adjmat.CurrentMat, phi_2, vec_help);
  VecDot(vec_help, phi_1, &c_s);
  
  if(mpirank == 0){
    std::cout << "# Time    Cs" << std::endl;
    std::cout << "0.0000000" << " " << PetscRealPart(c_s) / l << std::endl; 
  }

  for(unsigned int tt = 1; tt < (iterations); ++tt){
    // First step: Compute evolution under Hamiltonian
    te.krylov_evo(times[tt], times[tt - 1], phi_1);
    te.krylov_evo(times[tt], times[tt - 1], phi_2);

    // Compute the expectation value
    MatMult(adjmat.CurrentMat, phi_2, vec_help);
    VecDot(vec_help, phi_1, &c_s);

    if(mpirank == 0)
      std::cout << times[tt] << " " << PetscRealPart(c_s) / l << std::endl;
  }

  VecDestroy(&initial);
  VecDestroy(&phi_1);
  VecDestroy(&phi_2);
  VecDestroy(&vec_help);

  return 0;
}
