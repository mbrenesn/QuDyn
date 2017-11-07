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
  double tau = 0.777;
  double delta = 0.777;
  double h = 0.777;
  double t = 0.777;
  double max_time = 0.777;
  unsigned int time_its = 777;

  if(argc < 7){
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " -l [sites] -n [fill] -tau [tau] -delta [delta] -z [h] -t [t] -max_time [tmax] -time_its [its] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "-l") l = atoi(argv[i + 1]);
    else if(str == "-n") n = atoi(argv[i + 1]);
    else if(str == "-tau") tau = atof(argv[i + 1]);
    else if(str == "-delta") delta = atof(argv[i + 1]);
    else if(str == "-z") h = atof(argv[i + 1]);
    else if(str == "-t") t = atof(argv[i + 1]);
    else if(str == "-max_time") max_time = atof(argv[i + 1]);
    else if(str == "-time_its") time_its = atoi(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || tau == 0.777 || delta == 0.777 || h == 0.777 || t == 0.777 || 
          max_time == 0.777 || time_its == 777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: mpirun -np <proc> " << argv[0] << 
      " -l [sites] -n [fill] -tau [tau] -delta [delta] -z [h] -t [t] -max_time [tmax] -time_its [its] -[PETSc/SLEPc options]" << std::endl;
    exit(1);
  }

  std::vector<double> h_vec(l, 0.0);
  int pos = l / 2;
  h_vec[pos] = h;
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

  AdjacencyOp adjmat(env, *basis, t); 

  DiagonalOp diagop(env, *basis, true);
  diagop.construct_xxz_diagonal(basis->int_basis, tau, delta, h_vec);

  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  Vec initial;
  VecDuplicate(diagop.DiagonalVec, &initial);
  VecZeroEntries(initial);
  VecSetValue(initial, 0, 1.0, INSERT_VALUES);
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
