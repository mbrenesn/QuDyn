#include "ChebyshevEvo.h"

ChebyshevEvo::ChebyshevEvo(const double &tol,
                           const int &max_its,
                           const double &alpha)
: Emin(0), Emax(0)
{
  alpha_ = alpha;
  max_its_ = max_its;
  tol_ = tol;
}

ChebyshevEvo::~ChebyshevEvo()
{
  EPSDestroy(&eps_);
}

void ChebyshevEvo::initialize(Mat &ham_mat)
{
  // Get Emin and Emax
  EPSConvergedReason reason;
  EPSCreate(PETSC_COMM_WORLD, &eps_);
  EPSSetProblemType(eps_, EPS_HEP);
      
  EPSSetOperators(eps_, ham_mat, NULL);

  EPSSetWhichEigenpairs(eps_, EPS_LARGEST_REAL);
  EPSSolve(eps_);

  EPSGetConvergedReason(eps_, &reason);
  if(reason < 0){
    std::cerr << "EPS solver did not converge" << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  PetscScalar dummy;
  EPSGetEigenvalue(eps_, 0, &Emax, &dummy);

  EPSSetWhichEigenpairs(eps_, EPS_SMALLEST_REAL);
  EPSSolve(eps_);
  
  EPSGetConvergedReason(eps_, &reason);
  if(reason < 0){
    std::cerr << "EPS solver did not converge" << std::endl;
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }
  
  EPSGetEigenvalue(eps_, 0, &Emin, &dummy);

  // Scale values
  b_scale = 0.5 * (Emax + Emin);
  epsilon = alpha_ * (Emax - Emin);
  a_scale = 0.5 * (Emax - Emin + epsilon);

  // Rescale Hamiltonian
  MatShift(ham_mat, -1.0 * b_scale);
  MatScale(ham_mat, 1.0 / a_scale);
}

void ChebyshevEvo::chebyshev_evo(const double &final_time,
                                 const double &initial_time,
                                 Vec &vec,
                                 const Mat &ham_mat)
{
  PetscReal Emin_r = PetscRealPart(Emin);
  PetscReal Emax_r = PetscRealPart(Emax);
  PetscReal epsilon_r = PetscRealPart(epsilon);

  PetscReal argument = (Emax_r - Emin_r + epsilon_r) * (final_time - initial_time) * 0.5;
  PetscScalar phase_factor = PetscExpComplex(1.0 * PETSC_i * final_time * b_scale); 

  PetscReal minimum_rank = PetscCeilReal( (Emax_r - Emin_r) * (final_time - initial_time) * 0.5);

  if(minimum_rank < 3){
    minimum_rank = 3;
  }
 
  // Bessel coefficients
  std::vector<PetscScalar> chev_coeff;
  PetscReal chev_val = boost::math::cyl_bessel_j(0, argument);
  chev_coeff.push_back( chev_val * phase_factor);
  for(unsigned int it = 1; it < minimum_rank; ++it){
    chev_val = boost::math::cyl_bessel_j(it, argument);
    chev_coeff.push_back( 2 * PetscPowComplex(PETSC_i, it) * chev_val * phase_factor);
  }

  // Procedure
  Vec phi_0, phi_1, phi_2;
  VecDuplicate(vec, &phi_0);
  VecDuplicate(vec, &phi_1);
  VecDuplicate(vec, &phi_2);

  VecCopy(vec, phi_0);
  MatMult(ham_mat, phi_0, phi_1);
  VecAXPBYPCZ(vec, chev_coeff[0], chev_coeff[1], 0.0, phi_0, phi_1);

  unsigned int last_it = 0;
  for(unsigned int it = 2; it < minimum_rank; ++it){
    VecScale(phi_0, -1.0);
    VecScale(phi_1, 2.0);
    MatMultAdd(ham_mat, phi_1, phi_0, phi_2);
    VecAXPBY(vec, chev_coeff[it], 1.0, phi_2);
    VecScale(phi_1, 0.5);
    VecCopy(phi_1, phi_0);
    VecCopy(phi_2, phi_1);
    last_it = it;
  }

  double l2_norm;
  while(true){
    last_it++;
    chev_val = boost::math::cyl_bessel_j(last_it, argument);
    chev_coeff.push_back( 2 * PetscPowComplex(PETSC_i, last_it) * chev_val * phase_factor);
    VecScale(phi_0, -1.0);
    VecScale(phi_1, 2.0);
    MatMultAdd(ham_mat, phi_1, phi_0, phi_2);
    VecAXPBY(vec, chev_coeff[last_it], 1.0, phi_2);
    VecScale(phi_1, 0.5);
    VecCopy(phi_1, phi_0);
    VecCopy(phi_2, phi_1);
   
    if(last_it % 3 == 0){
      VecNorm(vec, NORM_2, &l2_norm);
      if(PetscAbsReal(l2_norm - 1.0) < tol_){
        break;
      }
    }
    if(last_it == max_its_){
      std::cout << "No convergence from Chebyshev up to max its" << std::endl;
      break;
    }
  }

  VecDestroy(&phi_0);
  VecDestroy(&phi_1);
  VecDestroy(&phi_2);
}
