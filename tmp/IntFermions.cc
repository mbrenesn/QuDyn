#include "IntFermions.h"

IntFermions::IntFermions(const Environment &env, 
                         const Basis &basis)
: hmag(NULL)
{
  l_ = env.l;
  n_ = env.n;
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  nlocal_ = basis.nlocal;
  start_ = basis.start;
  end_ = basis.end;
  basis_size_ = basis.basis_size;
}

/*******************************************************************************/
// Copy constructor
/*******************************************************************************/
IntFermions::IntFermions(const IntFermions &rhs)
{
  l_ = rhs.l_;
  n_ = rhs.n_;
  mpirank_ = rhs.mpirank_;
  mpisize_ = rhs.mpisize_;
  nlocal_ = rhs.nlocal_;
  start_ = rhs.start_;
  end_ = rhs.end_;
  basis_size_ = rhs.basis_size_;
  hmag = NULL;
}

/*******************************************************************************/
// Assignment operator
/*******************************************************************************/
IntFermions &IntFermions::operator=(const IntFermions &rhs)
{
  VecDestroy(&hmag);
  l_ = rhs.l_;
  n_ = rhs.n_;
  mpirank_ = rhs.mpirank_;
  mpisize_ = rhs.mpisize_;
  nlocal_ = rhs.nlocal_;
  start_ = rhs.start_;
  end_ = rhs.end_;
  basis_size_ = rhs.basis_size_;
  hmag = NULL;

  return *this;
}

IntFermions::~IntFermions()
{
  VecDestroy(&hmag);
}

/*******************************************************************************/
// Collects the nonlocal start index (global indices) of every processor, to be used
// during the construction of the matrix
/*******************************************************************************/
void IntFermions::gather_nonlocal_values_(LLInt *start_inds)
{
  MPI_Allgather(&start_, 1, MPI_LONG_LONG, start_inds, 1, MPI_LONG_LONG, 
    PETSC_COMM_WORLD);
}

/*******************************************************************************/
// Determines the sparsity pattern to allocate memory only for the non-zero 
// entries of the matrix
/*******************************************************************************/
void IntFermions::determine_allocation_details_(LLInt *int_basis, 
                                                std::vector<LLInt> &cont, 
                                                std::vector<LLInt> &st, 
                                                PetscInt *diag, 
                                                PetscInt *off)
{
  for(PetscInt i = 0; i < nlocal_; ++i) diag[i] = 1;

  for(PetscInt state = start_; state < end_; ++state){

    boost::dynamic_bitset<> bs(l_, int_basis[state - start_]);

    // Loop over all sites of the bit representation
    for(unsigned int site = 0; site < (l_ - 1); ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        int next_site1 = (site + 1);

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          LLInt new_int1 = Utils::binary_to_int(bitset, l_);
          // Loop over all states and look for a match
          LLInt match_ind1 = Utils::binsearch(int_basis, nlocal_, new_int1); 
          if(match_ind1 == -1){
            cont.push_back(new_int1);
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
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          LLInt new_int0 = Utils::binary_to_int(bitset, l_);
          // Loop over all states and look for a match
          LLInt match_ind0 = Utils::binsearch(int_basis, nlocal_, new_int0); 
          if(match_ind0 == -1){
            cont.push_back(new_int0);
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
  
  //std::cout << "Cont from Proc " << mpirank_ << std::endl;
  //for(LLInt i = 0; i < cont_size; ++i) std::cout << cont[i] << std::endl;

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
// Computes the Hamiltonian matrix given by means of the integer basis
/*******************************************************************************/
void IntFermions::construct_intfermions_hamiltonian(Mat &ham_mat, 
                                                    LLInt *int_basis, 
                                                    double V, 
                                                    double t, 
                                                    bool sites) 
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
  MatMPIAIJSetPreallocation(ham_mat, 0, d_nnz, 0, o_nnz);

  PetscFree(d_nnz);
  PetscFree(o_nnz);

  // Magnetization measurements
  if(sites){
    VecCreate(PETSC_COMM_WORLD, &hmag);
    VecSetSizes(hmag, nlocal_, basis_size_);
    VecSetType(hmag, VECMPI);
  }
  
  // Hamiltonian matrix construction
  PetscScalar ti = t;
  PetscScalar Vi = V;
  PetscScalar zero = 0.0;

  // Grab 1 of the states and turn it into bit representation
  for(PetscInt state = start_; state < end_; ++state){
    
    boost::dynamic_bitset<> bs(l_, int_basis[state - start_]);

    std::vector<double> spins(l_);
    for(unsigned int i = 0; i < l_; ++i)
      bs[i] == 1 ? spins[i] = 1.0 : spins[i] = -1.0; 

    if(sites){
      double site_half_mag = 0.5 * (spins[(l_ / 2) - 1] + 1);
      VecSetValue(hmag, state, site_half_mag, INSERT_VALUES);
    }

    // Loop over all sites of the bit representation
    PetscScalar int_term = 0.0;
    for(unsigned int site = 0; site < (l_ - 1); ++site){
      // A copy to avoid modifying the original basis
      boost::dynamic_bitset<> bitset = bs;
      
      // Case 1: There's a particle in this site
      if(bitset[site] == 1){
        
        int next_site1 = (site + 1);

        // If there's a particle in next site, do nothing
        if(bitset[next_site1] == 1){
          int_term += Vi;
          continue;
        }
        // Otherwise do a swap
        else{
          bitset[next_site1] = 1;
          bitset[site]       = 0;

          LLInt new_int1 = Utils::binary_to_int(bitset, l_);
          // Loop over all states and look for a match
          LLInt match_ind1 = Utils::binsearch(int_basis, nlocal_, new_int1); 
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

          MatSetValues(ham_mat, 1, &match_ind1, 1, &state, &ti, ADD_VALUES);
        }
      }      
      // Case 2: There's no particle in this site
      else{
        int next_site0 = (site + 1) % l_;

        // If there's a particle in the next site, a swap can occur
        if(bitset[next_site0] == 1){
          bitset[next_site0] = 0;
          bitset[site]       = 1;

          LLInt new_int0 = Utils::binary_to_int(bitset, l_);
          // Loop over all states and look for a match
          LLInt match_ind0 = Utils::binsearch(int_basis, nlocal_, new_int0); 
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
        
          MatSetValues(ham_mat, 1, &match_ind0, 1, &state, &ti, ADD_VALUES);
        }
        // Otherwise do nothing
        else{
          continue;
        }
      }    
    }

    MatSetValues(ham_mat, 1, &state, 1, &state, &int_term, ADD_VALUES);

    spins.erase(spins.begin(), spins.begin() + l_);
  }

  // Now cont contains the missing indices
  for(ULLInt in = 0; in < cont.size(); ++in){
    LLInt st_c = st[in];
    LLInt cont_c = cont[in];
    MatSetValues(ham_mat, 1, &cont_c, 1, &st_c, &ti, ADD_VALUES);
  }

  if(sites){
    VecAssemblyBegin(hmag);
    VecAssemblyEnd(hmag);
  }

  MatAssemblyBegin(ham_mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ham_mat, MAT_FINAL_ASSEMBLY);

  MatSetOption(ham_mat, MAT_SYMMETRIC, PETSC_TRUE);
}
