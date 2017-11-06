#include "DiagonalOp.h"

/*******************************************************************************/
// Single custom constructor for this class.
// Creates the diagonal part of the Hamiltonian represented as parallel vector.
/*******************************************************************************/
DiagonalOp::DiagonalOp(const Environment &env,
                       const Basis &basis,
                       bool sigma_z_mats)
{
  l_ = env.l;
  n_ = env.n;
  mpirank_ = env.mpirank;
  mpisize_ = env.mpisize;
  nlocal_ = basis.nlocal;
  start_ = basis.start;
  end_ = basis.end;
  basis_size_ = basis.basis_size;
  sigma_z_mats_ = sigma_z_mats;

  VecCreateMPI(PETSC_COMM_WORLD, nlocal_, basis_size_, &DiagonalVec);
  if(sigma_z_mats_){
    SigmaZ.resize(l_);
    for(unsigned int i = 0; i < SigmaZ.size(); ++i){
      VecDuplicate(DiagonalVec, &SigmaZ[i]);
    }
  }
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
  sigma_z_mats_ = rhs.sigma_z_mats_;

  VecDuplicate(rhs.DiagonalVec, &DiagonalVec);
  VecCopy(rhs.DiagonalVec, DiagonalVec);
  if(rhs.sigma_z_mats_){
    SigmaZ.resize(rhs.l_);
    for(unsigned int i = 0; i < SigmaZ.size(); ++i){
      VecDuplicate(rhs.SigmaZ[i], &SigmaZ[i]);
      VecCopy(rhs.SigmaZ[i], SigmaZ[i]);
    }
  }
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
    sigma_z_mats_ = rhs.sigma_z_mats_;

    VecDuplicate(rhs.DiagonalVec, &DiagonalVec);
    VecCopy(rhs.DiagonalVec, DiagonalVec);
    if(rhs.sigma_z_mats_){
      SigmaZ.resize(rhs.l_);
      for(unsigned int i = 0; i < SigmaZ.size(); ++i){
        VecDuplicate(rhs.SigmaZ[i], &SigmaZ[i]);
        VecCopy(rhs.SigmaZ[i], SigmaZ[i]);
      }
    }
  }

  return *this;
}

DiagonalOp::~DiagonalOp()
{
  VecDestroy(&DiagonalVec);
  if(sigma_z_mats_){
    for(unsigned int i = 0; i < SigmaZ.size(); ++i){
      VecDestroy(&SigmaZ[i]);
    }
  }
}

/*******************************************************************************/
// Constructs the diagonal part of XXZ Hamiltonian
/*******************************************************************************/
void DiagonalOp::construct_xxz_diagonal(LLInt *int_basis,
                                        double &tau,
                                        double &delta,
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
      if(bitset & (1 << site)){ 
          if(sigma_z_mats_) VecSetValue(SigmaZ[site], state, 1.0, INSERT_VALUES);
          mag_term += h[site] * tau * 0.5;
      }
      else{ 
          if(sigma_z_mats_) VecSetValue(SigmaZ[site], state, -1.0, INSERT_VALUES);
          mag_term -= h[site] * tau * 0.5;
      }

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
  VecAssemblyBegin(DiagonalVec);
  VecAssemblyEnd(DiagonalVec);
}

/*******************************************************************************/
// Computes the diagonal part of the Schwinger Hamiltonian or the diagonal
// of the superselection sector implementation
/*******************************************************************************/
void DiagonalOp::construct_schwinger_diagonal(LLInt *int_basis, 
                                              double &V, 
                                              double &h, 
                                              bool &rand)
{
  std::vector<unsigned int> Cnl(l_ * l_);
  std::vector<unsigned int> pre_fact(l_ - 1);
  // Cnl matrix
  for(unsigned int i = 0; i < l_; ++i){
    Cnl[i * l_ + i] = 0;
    for(unsigned int j = 0; j < l_; ++j){
      if(j > i) Cnl[i * l_ + j] = l_ - j - 1;
      else if(j < i) Cnl[i * l_ + j] = l_ - i - 1;
    }
  }

  // On-site term prefactor
  pre_fact[0] = l_ / 2;
  for(unsigned int i = 1; i < (l_ - 1); i += 2){
    pre_fact[i] = pre_fact[i - 1] - 1;
    pre_fact[i + 1] = pre_fact[i - 1] - 1;
  }

  // Random static charge configuration
  std::vector<int> rand_seq(l_ - 1);
  if(rand){
    if(mpirank_ == 0){
      boost::random::mt19937 gen;
      gen.seed(static_cast<LLInt>(std::time(0)));
      boost::random::uniform_int_distribution<LLInt> dist(-1, 1);
      for(unsigned int i = 0; i < (l_ - 1); ++i)
        rand_seq[i] = dist(gen);
    }
    MPI_Bcast(&rand_seq[0], (l_ - 1), MPI_INT, 0, PETSC_COMM_WORLD);
  }

  // Grab 1 of the states and turn it into bit representation
  for(PetscInt state = start_; state < end_; ++state){
    
    LLInt bs = int_basis[state - start_];

    std::vector<double> spins(l_);
    for(unsigned int i = 0; i < l_; ++i)
      bs & (1 << i) ? spins[i] = 1.0 : spins[i] = -1.0; 

    // On-site term and int term
    double os_term = 0.0; double os_term2 = 0.0; double mag_term = 0.0;
    double int_term = 0.0;
    for(unsigned int site = 0; site < l_; ++site){
      os_term += h * std::pow(-1, site + 1) * spins[site];
      
      if(site == (l_ - 1)) continue;

      if(rand){
        double sigma = 0.0; double q_i = 0.0;
        for(unsigned int i = 0; i < (site + 1); ++i){
          sigma += spins[i];
          q_i += rand_seq[i]; 
        }
        os_term2 += V * sigma * (q_i - ( (site + 1) % 2 ));
      }
      else 
        os_term2 += V * pre_fact[site] * spins[site]; // Schwinger

      for(unsigned int j = 0; j < (l_ - 1); ++j){
        int_term += V * spins[site] * spins[j] * Cnl[site * l_ + j];
      }
    }

    // Important! Cnl is counting twice!
    PetscScalar diag_term;
    if(rand) diag_term = os_term + os_term2 + (0.5 * int_term);
    else diag_term = os_term - os_term2 + (0.5 * int_term); // Schwinger
    VecSetValue(DiagonalVec, state, diag_term, INSERT_VALUES);

    spins.erase(spins.begin(), spins.begin() + l_);
  }
  VecAssemblyBegin(DiagonalVec);
  VecAssemblyEnd(DiagonalVec);
}
