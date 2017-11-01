#ifndef __KRYLOV_EVO_H
#define __KRYLOV_EVO_H

#include "../Environment/Environment.h"
#include "../Basis/Basis.h"

class KrylovEvo
{
  private:
    MFN mfn_; 
    FN f_;

  public:
    KrylovEvo(const Mat &ham_mat,
              const double &tol,
              const int &max_kryt_its);
    ~KrylovEvo();
    MFNConvergedReason reason;
    void krylov_evo(const double &final_time,
                    const double &initial_time,
                    Vec &vec);
};
#endif
