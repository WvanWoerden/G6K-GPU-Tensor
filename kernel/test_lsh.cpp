// AVX2 BDGL-like bucketer 
// Used by the cpu-only BDGL sieve, and the outdated gpu BDGL sieve.
// An independent, newer and documented version of this code 
// can be found at https://github.com/lducas/AVX2-BDGL-bucketer

#include <iostream>
#include <bitset>
#include "math.h"
#include "fht_lsh.h"

int N = 49;

void sample_spherical(int N, float *tab)
{
  float norm = 0;
  float corner_case = (rand()) / static_cast <float> (RAND_MAX);
  if (corner_case < 0.001)
  {
    for (int i = 0; i < N; ++i) tab[i] = 0;
    size_t j =  (static_cast<size_t> (rand())) % N;
    tab[j] = 1;
    return;
  }

  for (int i = 0; i < N; ++i) 
  {
    tab[i] =  (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5);
    tab[i] += (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5);
    tab[i] += (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5);
    tab[i] += (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5);
    tab[i] += (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5);
    tab[i] += (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5);
    norm += tab[i]*tab[i];
  }
  norm = 1/sqrt(norm);

  for (int i = 0; i < N; ++i) 
  {
    tab[i] *= -norm;
  }
}

float ip(int N, float *tab, float *tab0)
{
  double res = 0;
  for (int i = 0; i < N; ++i) 
  {
    res += tab[i]*tab0[i];
  }
  return res;
}

int main() {

  for (int N = 64; N < 128; N+=1)
  {

    for (int M = 1; M < 20; M+=15)
    {
      printf("\n\n");
      for (int X = 2; X < 4; X++)
      {

        int64_t samples = (1 << 15);
        int64_t buckets = M * pow(2,  .2015 *N);
        int64_t references = (1 << 14);

        ProductLSH lsh(N, X, buckets, M, 1);
        std::vector<std::vector<float> > ref(references, std::vector<float>(N));
        int32_t h[M]; 
        std::vector<std::vector<size_t> > buck(buckets);

        float v[N] = {0};

        double ip_pass = 0, ip_all = 0;
        int pass = 0;

        for (int i = 0; i < references; ++i)
        {
          sample_spherical(N, ref[i].data());


          lsh.hash(ref[i].data(), h);

          for (int j = 0; j < M; ++j)
          {
            assert(h[j] >= 0);
            assert(h[j] < buckets);

            buck[h[j]].push_back(i);
  
          }
        }

        for (int r = 0; r < samples; ++r)
        {
          sample_spherical(N, v);
          lsh.hash(v, h);

          float ipp = ip(N, v, ref[0].data());
          ip_all += ipp*ipp;

          for (int j = 0; j < M; ++j)
          {
            for (int k : buck[h[j]])
            {
              float ipp = ip(N, v, ref[k].data());
              ip_pass += ipp*ipp;
              pass ++;              
            }
          }
        }

        printf("\n dim %2d buckets %6d blocks %d multi %d \n", N,(int) buckets, X, M);
        printf("collisions ratio : %6d / %e = %f *expected \n", pass, (1.*references)*samples, (1.*buckets*pass) / (references*samples * M * M));
        printf("std-ip : collision %.4f  all %.4f \n", sqrt(ip_pass/pass) ,sqrt(ip_all/(samples)));
      }
    }
  }
  return 0;
}
