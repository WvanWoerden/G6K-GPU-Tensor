#ifndef G6K_SIMHASH_INL
#define G6K_SIMHASH_INL

#ifndef G6K_SIEVER_H
#error Do not include siever.inl directly
#endif

// choose the vectors sparse vectors r_i for the compressed representation
inline void SimHashes::reset_compress_pos(Siever const &siever)
{
    if( XPC_WORD_LEN == 0 )
        return;
    
    n = siever.n;
    if (n < 30)
    {
        for(size_t i = 0; i < XPC_BIT_LEN; ++i)
        {
            for(size_t j = 0; j < 6; ++j)
            {
                compress_pos[i][j] = 0;
            }
        }
        return;
    }

    size_t x, y;
    std::string const filename = siever.params.simhash_codes_basedir
      + "/spherical_coding/sc_"+std::to_string(n)+"_"+std::to_string(XPC_BIT_LEN)+".def";
    std::ifstream in(filename);
    std::vector<int> permut;

    // create random permutation of 0..n-1:
    permut.resize(n);
    std::iota(permut.begin(), permut.end(), 0);
    std::shuffle(permut.begin(), permut.end(), sim_hash_rng);

    if (!in)
    {
        std::string s = "Cannot open file ";
        s += filename;
        throw std::runtime_error(s);
    }

    for (y = 0; y < XPC_BIT_LEN; y++)
    {
        for (x = 0; x < 6; x++)
        {
            int v;
            in >> v;
            compress_pos[y][x] = permut[v];
        }
    }
    in.close();
}


// Compute the compressed representation of an entry
inline CompressedVector SimHashes::compress(std::array<LFT,YR_DIM> const &v) const
{
    if( XPC_WORD_LEN==0 )
        return {};

    ATOMIC_CPUCOUNT(260);
    CompressedVector c = {};
    if (n < 30) return c;
    for (size_t j = 0; j < XPC_WORD_LEN; ++j)
    {
        uint64_t c_tmp = 0;
        LFT a = 0;
        for (size_t i = 0; i < 64; i++)
        {
            size_t k = 64 * j + i;
            a   = v[compress_pos[k][0]];
            a  += v[compress_pos[k][1]];
            a  += v[compress_pos[k][2]];
            a  -= v[compress_pos[k][3]];
            a  -= v[compress_pos[k][4]];
            a  -= v[compress_pos[k][5]];

            c_tmp = c_tmp << 1;
            c_tmp |= (uint64_t)(a > 0);
        }
        c[j] = c_tmp;
    }
    return c;
}

void DualHashes::disable() {
    nr_vecs = 0;
    k = 0;
    acceptance_radius = 0;
}

void DualHashes::reset_dual_vecs(Siever const &siever, const std::vector<std::vector<LFT>> new_dual_vecs, float _conv_ratio, unsigned int _target_index) {
    conv_ratio = _conv_ratio;
    target_index = _target_index;

    nr_vecs = new_dual_vecs.size();
    assert( nr_vecs > 0 );
    assert( nr_vecs % 4 == 0 );
    k = new_dual_vecs[0].size();
    if( OTF_LIFT_HELPER_DIM > 0 and k > OTF_LIFT_HELPER_DIM ) {
        std::cerr << "Something went wrong.. " << siever.l << " " << siever.ll << std::endl;
    }
    
    //std::cout << "reset: " << nr_vecs << " " << k << " " << target_index << " " << siever.l << " " << siever.r << std::endl;

    // swap order because OTF_LIFT_HELPER_DIM 0,.., k-1 corresponds to l-1, ..., l-k in the context
    dual_vecs.resize(nr_vecs);
    for(size_t i = 0; i < nr_vecs; i++ ) {
        dual_vecs[i].resize(k);
        for(size_t j = 0; j < k; j++ ) {
            dual_vecs[i][j] = new_dual_vecs[i][k-1-j];
        }
    }

    ball_volume = std::pow(M_PI, k/2.) / tgamma( k/2. + 1);

    tests = 0;
    tests_passed = 0;

    // assume initial lenbound of 1.4, is updated in later iterations
    update_dh_bound( siever, 1.4 );
    }

inline void DualHashes::update_dh_bound( Siever const &siever, float lenbound ) {
    acceptance_radius = 0;
    if(nr_vecs > 0 ) {
        float bound = siever.get_lift_bound( target_index ) - lenbound * siever.gh;
        float dual_bound = conv_ratio * bound;
        dual_bound = std::max(dual_bound, float(0.));
        acceptance_radius = size_t(256*256*dual_bound);
    }
}

// squared radius
inline int DualHashes::radius_for_ratio( const double x ) const {
   return int(256*256 * nr_vecs / double(k) * std::pow((x/ball_volume), 2. / k)); 
}

inline void DualHashes::reset_acceptance_radius( const double x) {
    acceptance_radius = radius_for_ratio( x );
}

inline double DualHashes::get_acceptance_ratio() {
    return ball_volume * std::pow(double(acceptance_radius)/(256*256*nr_vecs/double(k)), k/2.);
}

inline DualHashVector DualHashes::compress(std::array<LFT, OTF_LIFT_HELPER_DIM> const &v) const {
    if( k > OTF_LIFT_HELPER_DIM )
        std::cerr << k << std::endl;
    assert( k <= OTF_LIFT_HELPER_DIM );
    
    DualHashVector dual_hash;
    for (size_t i = 0; i < nr_vecs; i++ ) {
        float sm = std::inner_product(v.cbegin(), v.cbegin()+k, dual_vecs[i].cbegin(), 0.0f);
        sm -= std::floor(sm);
        dual_hash[i] = uint8_t(sm * 256+0.5);
    }

    return dual_hash;
}

inline int DualHashes::sqdist(const DualHashVector &a, const DualHashVector &b, const bool sign) const {
    int dst = 0;
    
    if( sign ) {
        for( size_t i = 0; i < nr_vecs; i++ ) {
            uint8_t tmp = a[i] + b[i]; // overflow should handle modulo
            tmp = std::abs(static_cast<int8_t>(tmp));
            dst += int(tmp)*int(tmp);
        }
    }
    else {    
        for( size_t i = 0; i < nr_vecs; i++ ) {
            uint8_t tmp = a[i] - b[i]; // overflow should handle modulo
            tmp = std::abs(static_cast<int8_t>(tmp)); // acts as identity on [0..127] and as 256-x on [128...255]
            dst += int(tmp)*int(tmp);
        }
    }
    return dst;
}

inline int DualHashes::sqlen(const DualHashVector &a) const {
    int dst = 0;
    for( size_t i = 0; i < nr_vecs; i++ ) {
        uint8_t tmp = a[i]; // overflow should handle modulo
        tmp = std::abs(static_cast<int8_t>(tmp));
        dst += int(tmp)*int(tmp);
    }
    return dst;
}

inline bool DualHashes::test(const DualHashVector &a, const DualHashVector &b, const bool sign) {
    bool res = (sqdist(a,b,sign) <= acceptance_radius);
    tests++;
    tests_passed += res;
    return res;
}

inline LFT DualHashes::get_dual_vecs( int i, int j ) const {
    return dual_vecs[i][j];
}

inline unsigned int DualHashes::get_nr_vecs() const {
    return nr_vecs;
}

inline unsigned int DualHashes::get_dim_vecs() const {
    return k;
}

inline unsigned int DualHashes::get_dh_bound() const {
    return acceptance_radius;
}

#endif
