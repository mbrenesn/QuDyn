#ifndef __ENVIRONMENT_H
#define __ENVIRONMENT_H

#include <iostream>

#include <petscsys.h>
#include <slepcmfn.h>
#include <slepceps.h>

typedef PetscInt LLInt;

class Environment
{
  public:
    // Methods
    Environment(int argc, 
                char **argv, 
                unsigned int l, 
                unsigned int n);
    ~Environment();
    LLInt basis_size() const;
    void distribution(PetscInt b_size, 
                      PetscInt &nlocal, 
                      PetscInt &start, 
                      PetscInt &end) const;
    // Members
    unsigned int l, n;
    PetscMPIInt mpirank;
    PetscMPIInt mpisize;
  
  private:
};
#endif
