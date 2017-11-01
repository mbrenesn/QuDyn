#include "DiagonalOp.h"

/*******************************************************************************/
// Single custom constructor for this class.
// Creates the diagonal part of the Hamiltonian represented as parallel vector.
/*******************************************************************************/
DiagonalOp::DiagonalOp(const Environment &env,
                       const Basis &basis)
{
  l_ = env.l;
  n_ = env.n;
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  nlocal_ = basis.nlocal;
  start_ = basis.start;
  end_ = basis.end;
  basis_size_ = basis.basis_size;

  VecCreateMPI(PETSC_COMM_WORLD, nlocal_, basis_size_, &DiagonalVec);
}

/*******************************************************************************/
// Copy constructor
/*******************************************************************************/
DiagonalOp::DiagonalOp(const DiagonalOp &rhs)
{
  std::cout << "Copy constructor (diagonal op) has been called!" << std::endl;

  l_ = rhs.l_;
  n_ = rhs.n_;
  mpirank_ = rhs.mpirank_;
  mpisize_ = rhs.mpisize_;
  nlocal_ = rhs.nlocal_;
  start_ = rhs.start_;
  end_ = rhs.end_;
  basis_size_ = rhs.basis_size_;
  
  VecDuplicate(rhs.DiagonalVec, &DiagonalVec);
  VecCopy(rhs.DiagonalVec, DiagonalVec);
}

/*******************************************************************************/
// Assignment operator
/*******************************************************************************/
DiagonalOp &DiagonalOp::operator=(const DiagonalOp &rhs)
{
  std::cout << "Assignment operator (diagonal op) has been called!" << std::endl;
    
  if(this != &rhs){
    VecDestroy(&DiagonalVec);
      
    l_ = rhs.l_;
    n_ = rhs.n_;
    mpirank_ = rhs.mpirank_;
    mpisize_ = rhs.mpisize_;
    nlocal_ = rhs.nlocal_;
    start_ = rhs.start_;
    end_ = rhs.end_;
    basis_size_ = rhs.basis_size_;
    
    VecDuplicate(rhs.DiagonalVec, &DiagonalVec);
    VecCopy(rhs.DiagonalVec, DiagonalVec);
  }

  return *this;
}

DiagonalOp::~DiagonalOp()
{
  VecDestroy(&DiagonalVec);
}

/*******************************************************************************/
// Constructs the diagonal part of XXZ Hamiltonian
/*******************************************************************************/
void DiagonalOp::construct_xxz_diagonal(LLInt *int_basis,
                                        double tau,
                                        double delta,
                                        std::vector<double> &h) 
{
  // Grab 1 of the states and turn it into bit representation
  for(PetscInt state = start_; state < end_; ++state){
    
    LLInt bs = int_basis[state - start_];

    double Vi = 0.0; // Interaction part
    double mag_term = 0.0; // On-site term
    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < l_; ++site){
      // A copy to avoid modifying the original basis
      LLInt bitset = bs;
      
      // On-site term
      if(bitset & (1 << site)) mag_term += h[site] * tau * 0.5;
      else mag_term -= h[site] * tau * 0.5;

      // Open boundary coundition
      if(site == (l_ - 1)) continue;

      // Interaction
      // Case 1: There's a particle in this site
      if(bitset & (1 << site)){
        int next_site1 = (site + 1);

        // If there's a particle in next site, increase interaction
        if(bitset & (1 << next_site1)){
          Vi += delta * tau;
          continue;
        }
        // Otherwise decrease interaction
        else{
          Vi -= delta * tau;
          continue;
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1);

        // If there's a particle in the next site, decrease interaction
        if(bitset & (1 << next_site0)){
          Vi -= delta * tau;
          continue;
        }
        // Otherwise increase interaction
        else{
          Vi += delta * tau;
          continue;
        }
      }    
    }
    double diag_term = Vi + mag_term;
    VecSetValue(DiagonalVec, state, diag_term, INSERT_VALUES);
  }
}
