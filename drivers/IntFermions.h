#ifndef __INTFERMIONS_H
#define __INTFERMIONS_H

#include <stdexcept>
#include <ctime>

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"

class IntFermions
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
    IntFermions(const Environment &env, 
                const Basis &basis);
    ~IntFermions();
    IntFermions(const IntFermions &rhs);
    IntFermions &operator=(const IntFermions &rhs);
    Vec hmag;
    void construct_intfermions_hamiltonian(Mat &ham_mat, 
                                           LLInt *int_basis, 
                                           double V,
                                           double t, 
                                           bool sites); 
};
#endif
