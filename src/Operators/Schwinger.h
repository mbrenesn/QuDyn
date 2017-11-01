#ifndef __SCHWINGER_H
#define __SCHWINGER_H

#include <stdexcept>
#include <ctime>

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"

class Schwinger
{
  private:
    unsigned int l_, n_;
    PetscMPIInt mpirank_;
    PetscMPIInt mpisize_;
    LLInt basis_size_;
    PetscInt nlocal_;
    PetscInt start_;
    PetscInt end_;
    void gather_nonlocal_values_(LLInt *start_inds);
    void determine_allocation_details_(LLInt *int_basis, 
                                       std::vector<LLInt> &cont,
                                       std::vector<LLInt> &st, 
                                       PetscInt *diag, 
                                       PetscInt *off);

  public:
    Schwinger(const Environment &env, 
              const Basis &basis);
    ~Schwinger();
    Schwinger(const Schwinger &rhs);
    Schwinger &operator=(const Schwinger &rhs);
    Vec magnetization;
    Vec hmag;
    Vec hmag_1;
    Vec hmag_2;
    void construct_schwinger_hamiltonian(Mat &ham_mat, 
                                         LLInt *int_basis, 
                                         double V,
                                         double t, 
                                         double h, 
                                         bool magn, 
                                         bool sites, 
                                         bool rand);
};
#endif
