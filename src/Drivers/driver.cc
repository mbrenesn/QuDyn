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
  double J0 = 0.777;
  double Jz = 0.777;
  double F = 0.777;
  int periodic = -1;
  std::cout << std::fixed;
  std::cout.precision(8);

  if(argc < 13){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --j0 [j0] --jz [jz] --F [F] --periodic [0,1] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--l") l = atoi(argv[i + 1]);
    else if(str == "--n") n = atoi(argv[i + 1]);
    else if(str == "--j0") J0 = atof(argv[i + 1]);
    else if(str == "--jz") Jz = atof(argv[i + 1]);
    else if(str == "--F") F = atof(argv[i + 1]);
    else if(str == "--periodic") periodic = atoi(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || J0 == 0.777 || Jz == 0.777 || F == 0.777 || periodic == -1){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --j0 [j0] --jz [jz] --F [F] --periodic [0,1] -[PETSc/SLEPc options]" << std::endl;
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

  AdjacencyOp adjmat(env, *basis, J0, bc, false); 

  DiagonalOp diagop(env, *basis, bc, true, false);
  diagop.construct_starkm_diagonal(basis->int_basis, Jz, F);

  // Now put the effective Hamiltonian in AdjacencyMat
  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  //MatView(adjmat.AdjacencyMat, PETSC_VIEWER_STDOUT_WORLD);

  delete basis;

  // Time Evo
  double tol = 1.0e-7;
  double maxits = 1000000;
  int iterations = 1000 + 1;
  KrylovEvo te(adjmat.AdjacencyMat, tol, maxits);

  std::vector<double> times = linspace(0.0, 7, iterations);

  // Initial state
  Vec initial;
  VecDuplicate(diagop.DiagonalVec, &initial);
  VecZeroEntries(initial);
  LLInt index = l / 2;
  VecSetValue(initial, index, 1.0, INSERT_VALUES);
  VecAssemblyBegin(initial);
  VecAssemblyEnd(initial);

  // Initial values
  PetscScalar z_mag;
  Vec mag_help;
  VecDuplicate(initial, &mag_help);
  for(unsigned int i = 0; i < l; ++i){
    VecPointwiseMult(mag_help, diagop.SigmaZ[i], initial);
    VecDot(mag_help, initial, &z_mag);
    if(mpirank == 0)
      std::cout << (i + 1) << " " << "0.0" << " " << PetscRealPart(z_mag) << std::endl;
  }
  if(mpirank == 0)
    std::cout << std::endl;

  // Time evo
  for(unsigned int tt = 1; tt < (iterations); ++tt){
    te.krylov_evo(times[tt], times[tt - 1], initial);
    for(unsigned int i = 0; i < l; ++i){
      VecPointwiseMult(mag_help, diagop.SigmaZ[i], initial);
      VecDot(mag_help, initial, &z_mag);
      if(mpirank == 0)
        std::cout << (i + 1) << " " << times[tt] << " " << PetscRealPart(z_mag) << std::endl;
    }
    if(mpirank == 0)
      std::cout << std::endl;
  }

  VecDestroy(&initial);
  VecDestroy(&mag_help);

  return 0;
}
