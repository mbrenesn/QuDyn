#ifndef __ADJACENCYOP_H
#define __ADJACENCYOP_H

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"

class AdjacencyOp
{
  public:
    // Methods
    AdjacencyOp(const Environment &env,
                const Basis &basis,
                double t);
    ~AdjacencyOp();
    AdjacencyOp(const AdjacencyOp &rhs);
    AdjacencyOp &operator=(const AdjacencyOp &rhs);
    // Members
    Mat AdjacencyMat;
  
  private:
    unsigned int l_, n_;
    double t_;
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
    void construct_adjacency_(LLInt *int_basis);
};
#endif
