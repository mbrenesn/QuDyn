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
  int periodic = -1;
  double epsilon = 0.777;
  std::cout << std::fixed;
  std::cout.precision(8);

  if(argc < 7){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --periodic [0,1] --epsilon [e] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--l") l = atoi(argv[i + 1]);
    else if(str == "--n") n = atoi(argv[i + 1]);
    else if(str == "--alpha") alpha = atof(argv[i + 1]);
    else if(str == "--delta") delta = atof(argv[i + 1]);
    else if(str == "--h") h = atof(argv[i + 1]);
    else if(str == "--periodic") periodic = atoi(argv[i + 1]);
    else if(str == "--epsilon") epsilon = atof(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || alpha == 0.777 || delta == 0.777 || h == 0.777
    || periodic == -1 || epsilon == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --periodic [0,1] --epsilon [e] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  // BC
  bool bc = false;
  if(periodic == 1) bc = true; 
  // Field
  std::vector<double> h_vec(l, 0.0);
  h_vec[l / 2] = h;

  PetscLogDouble time_i, time_f, time_1, time_2;

  // Establish the environment
  Environment env(argc, argv, l, n);
  PetscTime(&time_i);

  PetscMPIInt mpirank = env.mpirank;
  PetscMPIInt mpisize = env.mpisize;

  if(mpirank == 0){
    std::cout << "# L = " << l << std::endl;
    std::cout << "# N = " << n << std::endl;
    std::cout << "# Alpha = " << alpha << std::endl;
    std::cout << "# Delta = " << delta << std::endl;
    std::cout << "# h = " << h << std::endl;
  }
  // Establish the basis environment, by pointer, to call an early destructor and reclaim
  // basis memory
  Basis *basis = new Basis(env);

  // Construct basis
  basis->construct_int_basis();
  //basis->print_basis(env);

  AdjacencyOp adjmat(env, *basis, alpha, bc, false); 

  DiagonalOp diagop(env, *basis, bc, true, false);
  diagop.construct_xxz_diagonal(basis->int_basis, delta, h_vec);

  // Now put the effective Hamiltonian in AdjacencyMat
  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  // Get extremal values
  EPS eps;
  PetscScalar k0, k1, Emin, Emax, dummy;
  EPSConvergedReason reason;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetProblemType(eps, EPS_HEP);
  
  EPSSetOperators(eps, adjmat.AdjacencyMat, NULL);
  
  PetscTime(&time_1);
  if(mpirank == 0){
    std::cout << "# Solving Emax... " << std::endl;
  }
  EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
  EPSSolve(eps);

  EPSGetConvergedReason(eps, &reason);
  if(reason < 0){
    std::cerr << "EPS solver did not converge LARGE EIGEN" << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  EPSGetEigenvalue(eps, 0, &k1, &dummy);
  Emax = PetscRealPart(k1);
  PetscTime(&time_2);
  if(mpirank == 0){
    std::cout << "# Time Emax = " << time_2 - time_1 << std::endl;
  }
 
  PetscTime(&time_1);
  if(mpirank == 0){
    std::cout << "# Solving Emin... " << std::endl;
  }
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSolve(eps);

  EPSGetConvergedReason(eps, &reason);
  if(reason < 0){
    std::cerr << "EPS solver did not converge SMALL EIGEN" << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  EPSGetEigenvalue(eps, 0, &k0, &dummy);
  Emin = PetscRealPart(k0);
  PetscTime(&time_2);
  if(mpirank == 0){
    std::cout << "# Time Emin = " << time_2 - time_1 << std::endl;
  }
    
  if(mpirank == 0){
    std::cout << "# Emin = " << Emin << " " << "Emax = " << Emax << std::endl;
  }

  EPSDestroy(&eps);

  //MatScale(adjmat.AdjacencyMat, 1.0 / (Emax - Emin));
  //MatShift(adjmat.AdjacencyMat, -1.0 * (Emin / (Emax - Emin)));
  
  // Mid-spectrum eigenvalues
  PetscScalar target = (epsilon * (Emax - Emin)) + Emin;
  EPS eps_si;
  EPSCreate(PETSC_COMM_WORLD, &eps_si);

  EPSSetProblemType(eps_si, EPS_HEP);

  EPSSetOperators(eps_si, adjmat.AdjacencyMat, NULL);

  EPSSetFromOptions(eps_si);
  EPSSetWhichEigenpairs(eps_si, EPS_TARGET_REAL);
  EPSSetTarget(eps_si, target);

  PetscTime(&time_1);
  if(mpirank == 0){
    std::cout << "# Solving spectra... " << std::endl;
  }
  EPSSolve(eps_si);
  PetscTime(&time_2);
  
  if(mpirank == 0){
    std::cout << "# Time spectra = " << time_2 - time_1 << std::endl;
  }
  
  Vec eigvec, tmp;
  VecCreateMPI(PETSC_COMM_WORLD, basis->nlocal, basis->basis_size, &eigvec);
  VecDuplicate(eigvec, &tmp);

  PetscInt nconv;
  EPSGetConverged(eps_si, &nconv);
  PetscScalar eigr;
  PetscReal e;
  PetscScalar sigma_zi;
  
  PetscTime(&time_1);
  if(mpirank == 0){
    std::cout << "# Computing mags... " << std::endl;
  }
  for(PetscInt i = 0; i < nconv; ++i){
    EPSGetEigenpair(eps_si, i, &eigr, NULL, eigvec, NULL);
    VecPointwiseMult(tmp, diagop.SigmaZ[l / 2], eigvec);
    VecDot(tmp, eigvec, &sigma_zi);
    e = PetscRealPart((eigr - Emin) / (Emax - Emin));
    if(mpirank == 0)
      std::cout << e << " " << PetscRealPart(sigma_zi) << std::endl; 
  }
  PetscTime(&time_2);
  
  if(mpirank == 0){
    std::cout << "# Time mags = " << time_2 - time_1 << std::endl;
  }

  delete basis;
  VecDestroy(&eigvec);
  VecDestroy(&tmp);
  EPSDestroy(&eps_si);
  PetscTime(&time_f);

  if(mpirank == 0) std::cout << "# Total time: " << time_f - time_i << std::endl; 

  return 0;
}
