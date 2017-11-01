#ifndef __BASIS_H
#define __BASIS_H

#include <boost/dynamic_bitset.hpp>

#include "../Environment/Environment.h"

class Basis
{
  public:
    // Methods
    Basis(const Environment &env);
    ~Basis();
    Basis(const Basis &rhs);
    Basis &operator=(const Basis &rhs);
    void construct_int_basis();
    void print_basis(const Environment &env, 
                     bool bits = false);
    void construct_bit_basis(boost::dynamic_bitset<> *bit_basis);
    // Members
    LLInt basis_size;
    PetscInt basis_local;
    PetscInt basis_start;
    PetscInt nlocal;
    PetscInt start;
    PetscInt end;
    LLInt *int_basis;
  
  private:
    unsigned int l_, n_;
    LLInt factorial_(LLInt n);
    LLInt first_int_();
};
#endif
