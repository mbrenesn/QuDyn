#ifndef __CHEBYSHEV_EVO_H
#define __CHEBYSHEV_EVO_H

#include <boost/math/special_functions/bessel.hpp>

#include "../Environment/Environment.h"
#include "../Basis/Basis.h"

class ChebyshevEvo
{
  private:
    EPS eps_; 
    double alpha_;
    int max_its_;
    double tol_;

  public:
    ChebyshevEvo(const double &tol,
                 const int &max_its,
                 const double &alpha);
    ~ChebyshevEvo();
    PetscScalar Emin;
    PetscScalar Emax;
    PetscScalar a_scale;
    PetscScalar b_scale;
    PetscScalar epsilon;
    void initialize(Mat &ham_mat);
    void chebyshev_evo(const double &final_time,
                       const double &initial_time,
                       Vec &vec,
                       const Mat &ham_mat);
};
#endif
