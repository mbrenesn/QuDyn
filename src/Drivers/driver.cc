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
  double J = 0.777;
  double V = 0.777;
  double gamma = 0.777;
  double alpha = 0.777;
  int periodic = -1;
  std::cout << std::fixed;
  std::cout.precision(8);

  if(argc < 15){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --J [J] --V [V] --gamma [gamma] --alpha [alpha] --periodic [0,1] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--l") l = atoi(argv[i + 1]);
    else if(str == "--n") n = atoi(argv[i + 1]);
    else if(str == "--J") J = atof(argv[i + 1]);
    else if(str == "--V") V = atof(argv[i + 1]);
    else if(str == "--gamma") gamma = atof(argv[i + 1]);
    else if(str == "--alpha") alpha = atof(argv[i + 1]);
    else if(str == "--periodic") periodic = atoi(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || J == 0.777 || V == 0.777 || gamma == 0.777 || alpha == 0.777 
      || periodic == -1){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --J [J] --V [V] --gamma [gamma] --alpha [alpha] --periodic [0,1] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  // BC
  bool bc = false;
  if(periodic == 1) bc = true; 

  PetscLogDouble time_i, time_f, time_1, time_2;

  // Establish the environment
  Environment env(argc, argv, l, n);
  PetscTime(&time_i);

  PetscMPIInt mpirank = env.mpirank;
  PetscMPIInt mpisize = env.mpisize;

  // Establish the basis environment, by pointer, to call an early destructor and reclaim
  // basis memory
  Basis *basis = new Basis(env);

  // Construct basis
  basis->construct_int_basis();
  //basis->print_basis(env);

  AdjacencyOp adjmat(env, *basis, J / 4.0, bc, false); 

  DiagonalOp diagop(env, *basis, bc, false, true);
  diagop.construct_starkm_diagonal(basis->int_basis, V, gamma, alpha);

  // Now put the effective Hamiltonian in AdjacencyMat
  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  // Time Evo
  double tol = 1.0e-7;
  double maxits = 1000000;
  int iterations = 1000 + 1;
  KrylovEvo te(adjmat.AdjacencyMat, tol, maxits);

  std::vector<double> times = linspace(0.0, 100, iterations);

  // Initial state
  InitialState init(env, *basis);
  init.neel_initial_state(basis->int_basis);

  Vec mag_help;
  VecDuplicate(init.InitialVec, &mag_help);

  PetscScalar imbalance;
  for(unsigned int tt = 1; tt < (iterations); ++tt){
    te.krylov_evo(times[tt], times[tt - 1], init.InitialVec);

    VecPointwiseMult(mag_help, diagop.TotalZ, init.InitialVec);
    VecDot(mag_help, init.InitialVec, &imbalance);

    if(mpirank == 0) std::cout << times[tt] << " " << PetscRealPart(imbalance) << std::endl;
  }

  VecDestroy(&mag_help);

  return 0;
}
