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

  PetscLogDouble time_i, time_f;

  // Establish the environment
  Environment env(argc, argv, l, n);
  PetscTime(&time_i);

  PetscMPIInt mpirank = env.mpirank;
  PetscMPIInt mpisize = env.mpisize;

  if(mpirank == 0){
    std::cout << "# Densities as a function of time" << std::endl;
    std::cout << "# L = " << l << std::endl;
    std::cout << "# N = " << n << std::endl;
    std::cout << "# J0 = " << J0 << std::endl;
    std::cout << "# JZ = " << Jz << std::endl;
    std::cout << "# F = " << F << std::endl;
    std::cout << "# Periodic: " << periodic << std::endl;
  }

  // Establish the basis environment, by pointer, to call an early destructor and reclaim
  // basis memory
  Basis *basis = new Basis(env);

  // Construct basis
  if(mpirank == 0) std::cout << "# Basis" << std::endl;
  basis->construct_int_basis();
  //basis->print_basis(env);
  if(mpirank == 0) std::cout << "# Basis done" << std::endl;

  if(mpirank == 0) std::cout << "# Adj mat" << std::endl;
  AdjacencyOp adjmat(env, *basis, J0, bc, false); 
  if(mpirank == 0) std::cout << "# Adj mat done" << std::endl;

  if(mpirank == 0) std::cout << "# Diag" << std::endl;
  DiagonalOp diagop(env, *basis, bc, true, false);
  diagop.construct_starkm_diagonal(basis->int_basis, Jz, F);
  if(mpirank == 0) std::cout << "# Diag done" << std::endl;

  // Now put the effective Hamiltonian in AdjacencyMat
  if(mpirank == 0) std::cout << "# Joining" << std::endl;
  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);
  if(mpirank == 0) std::cout << "# Joining done" << std::endl;

  if(mpirank == 0) std::cout << "# Constructed" << std::endl;
  if(mpirank == 0) std::cout << "# Procs: " << mpisize << std::endl;

  // Time Evo
  double tol = 1.0e-7;
  double maxits = 1000000;
  int iterations = 200 + 1;
  KrylovEvo te(adjmat.AdjacencyMat, tol, maxits);

  std::vector<double> times = linspace(0.0, 20.0, iterations);

  if(mpirank == 0) std::cout << "# Time Evo constructed" << std::endl;
  // Initial state: Typical
  Vec initial;
  VecDuplicate(diagop.DiagonalVec, &initial);

  boost::random::mt19937 gen;
  gen.seed(mpirank);
  int rtime = 0;
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

  // Initial state: 1 + (\sigma_z)^L/2
  // n_(L/2) vector  
  Vec nL2;
  VecDuplicate(diagop.DiagonalVec, &nL2);
  PetscScalar aa;
  for(PetscInt index = basis->start; index < basis->end; ++index){
    LLInt bs = basis->int_basis[index - (basis->start)];
    if(bs & (1 << (l / 2))) aa = 1.0;
    else aa = 0.0;
    VecSetValue(nL2, index, aa, INSERT_VALUES);
  }
  VecAssemblyBegin(nL2);
  VecAssemblyEnd(nL2);

  VecPointwiseMult(initial, nL2, initial);
  VecNorm(initial, NORM_2, &norm);
  VecNormalize(initial, &norm);

  delete basis;

  if(mpirank == 0) std::cout << "# Init vecs constructed" << std::endl;

  // Initial values
  PetscScalar z_mag;
  Vec mag_help;
  VecDuplicate(initial, &mag_help);
  for(unsigned int i = 0; i < l; ++i){
    VecPointwiseMult(mag_help, diagop.SigmaZ[i], initial);
    VecDot(mag_help, initial, &z_mag);
    if(mpirank == 0)
      std::cout << i << " " << "0.0" << " " << (PetscRealPart(z_mag) + 1.0) / 2.0 << std::endl;
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
        std::cout << i << " " << times[tt] << " " 
          << (PetscRealPart(z_mag) + 1.0) / 2.0 << std::endl;
    }
    if(mpirank == 0)
      std::cout << std::endl;
  }
  PetscTime(&time_f);

  if(mpirank == 0)
    std::cout << "# Time: " << time_f - time_i << std::endl;

  VecDestroy(&initial);
  VecDestroy(&nL2);
  VecDestroy(&mag_help);

  return 0;
}
