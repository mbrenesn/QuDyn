#include <iostream>
#include <sstream>

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"
#include "../Operators/AdjacencyOp.h"
#include "../Operators/DiagonalOp.h"

#include <petsctime.h>

int main(int argc, char **argv)
{
  int l = 4;
  int n = 2;
  double t = 2.0;
  double tau = 1.0;
  double delta = 1.0;
  std::vector<double> h(l, 1.0);
  h[1] = 2.0;
  h[2] = 3.0;
  h[3] = 4.0;
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

  DiagonalOp diagop(env, *basis);
  diagop.construct_xxz_diagonal(basis->int_basis, tau, delta, h);

  Utils::join_into_hamiltonian(adjmat.AdjacencyMat, diagop.DiagonalVec);

  MatView(adjmat.AdjacencyMat, PETSC_VIEWER_STDOUT_WORLD);

  delete basis;

  return 0;
}
