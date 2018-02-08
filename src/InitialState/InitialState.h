#ifndef __INITIAL_STATE_H
#define __INITIAL_STATE_H

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"

class InitialState
{
  public:
    // Methods
    InitialState(const Environment &env,
                 const Basis &basis);
    ~InitialState();
    InitialState(const InitialState &rhs);
    InitialState &operator=(const InitialState &rhs);
    // Members
    Vec InitialVec;
    void neel_initial_state(LLInt *int_basis);
    void random_initial_state(LLInt *int_basis,
                              bool wtime = false,
                              bool verbose = false);
    void purified_mixed_initial_state(LLInt *int_basis,
                                      double &mu,
                                      bool wtime = false);

  private:
    unsigned int l_, n_;
    PetscMPIInt mpirank_;
    PetscMPIInt mpisize_;
    LLInt basis_size_;
    PetscInt nlocal_;
    PetscInt start_;
    PetscInt end_;
};
#endif
