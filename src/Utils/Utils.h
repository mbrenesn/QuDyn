#ifndef __UTILS_H
#define __UTILS_H

#include <boost/dynamic_bitset.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cmath>

#include "../Environment/Environment.h"
#include "../Basis/Basis.h"

namespace Utils
{
  LLInt mod(LLInt a, LLInt b);
  LLInt binary_to_int(boost::dynamic_bitset<> bs,
                      unsigned int l);
  LLInt binsearch(const LLInt *array, 
                  LLInt len, 
                  LLInt value);
  LLInt get_neel_index(const Environment &env, 
                       const Basis &bas);
  LLInt get_random_index(const Environment &env,
                         const Basis &bas, 
                         bool wtime, 
                         bool verbose);
  void join_into_hamiltonian(Mat &mat, Vec &vec);
}
#endif
