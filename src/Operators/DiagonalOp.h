#ifndef __DIAGONALOP_H
#define __DIAGONALOP_H

#include "../Environment/Environment.h"
#include "../Utils/Utils.h"
#include "../Basis/Basis.h"

class DiagonalOp
{
  public:
    // Methods
    DiagonalOp(const Environment &env,
               const Basis &basis,
               bool sigma_z_mats = false);
    ~DiagonalOp();
    DiagonalOp(const DiagonalOp &rhs);
    DiagonalOp &operator=(const DiagonalOp &rhs);
    // Members
    Vec DiagonalVec;
    std::vector<Vec> SigmaZ;
    void construct_xxz_diagonal(LLInt *int_basis,
                                double &delta,
                                std::vector<double> &h);
    void construct_schwinger_diagonal(LLInt *int_basis,
                                      double &V,
                                      double &h,
                                      bool &rand);

  private:
    unsigned int l_, n_;
    PetscMPIInt mpirank_;
    PetscMPIInt mpisize_;
    LLInt basis_size_;
    PetscInt nlocal_;
    PetscInt start_;
    PetscInt end_;
    bool sigma_z_mats_;
};
#endif
