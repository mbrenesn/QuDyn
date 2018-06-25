#include "AdjacencyOp.h"

/*******************************************************************************/
// Single custom constructor for this class.
// Creates the adjacency matrix member depending on the basis chosen.
/*******************************************************************************/
AdjacencyOp::AdjacencyOp(const Environment &env,
                         const Basis &basis,
                         double t)
{
  l_ = env.l;
  n_ = env.n;
  t_ = t;
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  nlocal_ = basis.nlocal;
  start_ = basis.start;
  end_ = basis.end;
  basis_size_ = basis.basis_size;

  MatCreate(PETSC_COMM_WORLD, &AdjacencyMat);
  MatSetSizes(AdjacencyMat, nlocal_, nlocal_, basis_size_, basis_size_);
  MatSetType(AdjacencyMat, MATMPIAIJ);

  construct_adjacency_(basis.int_basis);
}

/*******************************************************************************/
// Copy constructor
/*******************************************************************************/
AdjacencyOp::AdjacencyOp(const AdjacencyOp &rhs)
{
  std::cout << "Copy constructor (adj matrix) has been called!" << std::endl;

  l_ = rhs.l_;
  n_ = rhs.n_;
  t_ = rhs.t_;
  mpirank_ = rhs.mpirank_;
  mpisize_ = rhs.mpisize_;
  nlocal_ = rhs.nlocal_;
  start_ = rhs.start_;
  end_ = rhs.end_;
  basis_size_ = rhs.basis_size_;
  
  MatDuplicate(rhs.AdjacencyMat, MAT_COPY_VALUES, &AdjacencyMat);
}

/*******************************************************************************/
// Assignment operator
/*******************************************************************************/
AdjacencyOp &AdjacencyOp::operator=(const AdjacencyOp &rhs)
{
  std::cout << "Assignment operator (adj matrix) has been called!" << std::endl;
    
  if(this != &rhs){
    MatDestroy(&AdjacencyMat);

    l_ = rhs.l_;
    n_ = rhs.n_;
    t_ = rhs.t_;
    mpirank_ = rhs.mpirank_;
    mpisize_ = rhs.mpisize_;
    nlocal_ = rhs.nlocal_;
    start_ = rhs.start_;
    end_ = rhs.end_;
    basis_size_ = rhs.basis_size_;
  
    MatDuplicate(rhs.AdjacencyMat, MAT_COPY_VALUES, &AdjacencyMat);
  }

  return *this;
}

AdjacencyOp::~AdjacencyOp()
{
  MatDestroy(&AdjacencyMat);
}

/*******************************************************************************/
// Collects the nonlocal start index (global indices) of every processor, to be used
// during the construction of the matrix
/*******************************************************************************/
void AdjacencyOp::gather_nonlocal_values_(LLInt *start_inds)
{
  MPI_Allgather(&start_, 1, MPI_LONG_LONG, start_inds, 1, MPI_LONG_LONG, 
    PETSC_COMM_WORLD);
}

/*******************************************************************************/
// Determines the sparsity pattern to allocate memory only for the non-zero 
// entries of the matrix. This matrix has zero values on diagonal so a Hamiltonian
// can be built from it.
/*******************************************************************************/
void AdjacencyOp::determine_allocation_details_(LLInt *int_basis, 
                                                std::vector<LLInt> &cont, 
                                                std::vector<LLInt> &st, 
                                                PetscInt *diag, 
                                                PetscInt *off)
{
  for(PetscInt i = 0; i < nlocal_; ++i) diag[i] = 1;

  for(PetscInt state = start_; state < end_; ++state){

    LLInt bs = int_basis[state - start_];

    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < (l_ - 1); ++site){
      // A copy to avoid modifying the original basis
      LLInt bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset & (1 << site)){
        int next_site1 = (site + 1);

        // If there's a particle in next site, do nothing
        if(bitset & (1 << next_site1)){
          continue;
        }
        // Otherwise do a swap
        else{
          bitset ^= 1 << site;
          bitset ^= 1 << next_site1;

          // Loop over all states and look for a match
          LLInt match_ind1 = Utils::binsearch(int_basis, nlocal_, bitset); 
          if(match_ind1 == -1){
            cont.push_back(bitset);
            st.push_back(state);
            continue;
          }
          else{
            match_ind1 += start_;
          }

          if(match_ind1 < end_ && match_ind1 >= start_) diag[state - start_]++;
          else off[state - start_]++; 
        }
      }
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1);

        // If there's a particle in the next site, a swap can occur
        if(bitset & (1 << next_site0)){
          bitset ^= 1 << site;
          bitset ^= 1 << next_site0;

          // Loop over all states and look for a match
          LLInt match_ind0 = Utils::binsearch(int_basis, nlocal_, bitset); 
          if(match_ind0 == -1){
            cont.push_back(bitset);
            st.push_back(state);
            continue;
          }
          else{
            match_ind0 += start_;
          }
          
          if(match_ind0 < end_ && match_ind0 >= start_) diag[state - start_]++;
          else off[state - start_]++;
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }    
    }
  }

  // Collective communication of global indices
  LLInt *start_inds = new LLInt[mpisize_];

  gather_nonlocal_values_(start_inds);

  // Proc 0 is always gonna have the larger section of the distribution (when rest is present)
  // so let's use this value as the size of the basis_help buffer
  LLInt basis_help_size;
  if(mpirank_ == 0) basis_help_size = nlocal_;

  MPI_Bcast(&basis_help_size, 1, MPI_LONG_LONG, 0, PETSC_COMM_WORLD);

  // Create basis_help buffers and initialize them to zero
  LLInt *basis_help = new LLInt[basis_help_size];
  for(LLInt i = 0; i < basis_help_size; ++i) basis_help[i] = 0;

  // At the beginning basis_help is just int_basis, with the remaining values set to zero
  // It's important that the array remains sorted for the binary lookup
  for(LLInt i = 0; i < nlocal_; ++i)
    basis_help[i + (basis_help_size - nlocal_)] = int_basis[i];

  // Main communication procedure. A ring exchange of the int_basis using basis_help memory
  // buffer, after a ring exchange occurs each processor looks for the missing indices of
  // the Hamiltonian and replaces the values of cont buffer
  PetscMPIInt next = (mpirank_ + 1) % mpisize_;
  PetscMPIInt prec = (mpirank_ + mpisize_ - 1) % mpisize_;

  LLInt cont_size = cont.size();

  for(PetscMPIInt exc = 0; exc < mpisize_ - 1; ++exc){
 
    MPI_Sendrecv_replace(&basis_help[0], basis_help_size, MPI_LONG_LONG, next, 0,
      prec, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);

    PetscMPIInt source = Utils::mod((prec - exc), mpisize_);
    for(LLInt i = 0; i < cont_size; ++i){
      if(cont[i] > 0){
        LLInt m_ind = Utils::binsearch(basis_help, basis_help_size, cont[i]);

        if(m_ind != -1){
          if(basis_help[0] == 0) m_ind = m_ind - 1;
          cont[i] = -1ULL * (m_ind + start_inds[source]);
        }
      }
    }
  }
  
  // Flip the signs
  for(PetscInt i = 0; i < cont_size; ++i) cont[i] = -1ULL * cont[i];
  
  // Now cont contains the missing indices
  for(LLInt in = 0; in < cont_size; ++in){
    LLInt st_c = st[in];
    if(cont[in] < end_ && cont[in] >= start_) diag[st_c - start_]++;
    else off[st_c - start_]++;
  }

  delete [] basis_help;
  delete [] start_inds;  
}

/*******************************************************************************/
// Constructs the adjacency matrix by means of the integer basis
/*******************************************************************************/
void AdjacencyOp::construct_adjacency_(LLInt *int_basis) 
{
  // Preallocation. For this we need a hint on how many non-zero entries the matrix will
  // have in the diagonal submatrix and the offdiagonal submatrices for each process

  // Allocating memory only for the non-zero entries of the matrix
  PetscInt *d_nnz, *o_nnz;
  PetscCalloc1(nlocal_, &d_nnz);
  PetscCalloc1(nlocal_, &o_nnz);

  std::vector<LLInt> cont;
  cont.reserve(basis_size_ / l_);
  std::vector<LLInt> st;
  st.reserve(basis_size_ / l_);
 
  determine_allocation_details_(int_basis, cont, st, d_nnz, o_nnz);

  // Preallocation step
  MatMPIAIJSetPreallocation(AdjacencyMat, 0, d_nnz, 0, o_nnz);

  PetscFree(d_nnz);
  PetscFree(o_nnz);

  PetscScalar ti = 2.0 * t_;
  PetscScalar dummy = 0.0;
  // Grab 1 of the states and turn it into bit representation
  for(PetscInt state = start_; state < end_; ++state){
    
    LLInt bs = int_basis[state - start_];
    MatSetValues(AdjacencyMat, 1, &state, 1, &state, &dummy, ADD_VALUES);

    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < (l_ - 1); ++site){
      // A copy to avoid modifying the original basis
      LLInt bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset & (1 << site)){
        int next_site1 = (site + 1);

        // If there's a particle in next site, do nothing
        if(bitset & (1 << next_site1)){
          continue;
        }
        // Otherwise do a swap
        else{
          bitset ^= 1 << site;
          bitset ^= 1 << next_site1;

          // Loop over all states and look for a match
          LLInt match_ind1 = Utils::binsearch(int_basis, nlocal_, bitset); 
          if(match_ind1 == -1){
            continue;
          }
          else{
            match_ind1 += start_;
          }
          
          if(match_ind1 == -1){
            std::cerr << "Error in the bin search within the Ham mat construction" << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
          } 

          MatSetValues(AdjacencyMat, 1, &match_ind1, 1, &state, &ti, ADD_VALUES);
        }
      }      
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1);

        // If there's a particle in the next site, a swap can occur
        if(bitset & (1 << next_site0)){
          bitset ^= 1 << site;
          bitset ^= 1 << next_site0;

          // Loop over all states and look for a match
          LLInt match_ind0 = Utils::binsearch(int_basis, nlocal_, bitset); 
          if(match_ind0 == -1){
            continue;
          }
          else{
            match_ind0 += start_;
          }
          
          if(match_ind0 == -1){
            std::cerr << "Error in the bin search within the Ham mat construction" << std::endl;
            MPI_Abort(PETSC_COMM_WORLD, 1);
          } 
        
          MatSetValues(AdjacencyMat, 1, &match_ind0, 1, &state, &ti, ADD_VALUES);
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }    
    }
  }

  // Now cont contains the missing indices
  for(ULLInt in = 0; in < cont.size(); ++in){
    LLInt st_c = st[in];
    LLInt cont_c = cont[in];
    MatSetValues(AdjacencyMat, 1, &cont_c, 1, &st_c, &ti, ADD_VALUES);
  }

  MatAssemblyBegin(AdjacencyMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(AdjacencyMat, MAT_FINAL_ASSEMBLY);

  MatSetOption(AdjacencyMat, MAT_SYMMETRIC, PETSC_TRUE);
}
