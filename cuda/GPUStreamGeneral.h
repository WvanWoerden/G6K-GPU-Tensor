#ifndef GPU_STREAM_H_
#define GPU_STREAM_H_

#ifdef HAVE_CUDA

#ifndef DEBUG_BENCHMARK
#define DEBUG_BENCHMARK 0
#endif // DEBUG_BENCHMARK

#define MAX_EXTEND 4
#define MAX_LIFT 32

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "../kernel/siever.h"

struct QEntry {
    size_t i,j;
    float len;
    int8_t sign;
};

template<size_t tuple_size>
struct Qtuple {
    uint32_t v[tuple_size];
    float len;
    // binary encoding of signs in tuple_size-1 most significant bits
    // v[0] + (-1)^(sign&1) * v[1] + ... + (-1)^(sign&(1<<(tuple_size-2))) * v[tuple_sign-1]
    uint8_t sign; 
};

template<size_t tuple_size>
struct Ltuple {
    uint32_t v[tuple_size];
    float len;
    uint8_t sign;
};

template<size_t tuple_size>
struct compare_QT
{
    bool operator()(Qtuple<tuple_size> const& lhs, Qtuple<tuple_size> const& rhs) const { return lhs.len < rhs.len; }
};

struct queues {
    std::deque<Qtuple<2>> sieve_pairs;
    std::deque<Qtuple<3>> sieve_triples;
    std::vector<Ltuple<2>> lift_pairs;
    std::vector<Ltuple<3>> lift_triples;
    size_t lifted_pairs = 0;
};

struct triple_bucket {
    indextype* indices;
    iptype* ips;
    uint32_t size = 0;
    indextype b_db_index;
    indextype b_local_index;
    lentype b_len;
};

struct global_dev_ptrs {
    float* dev_mu_L;
    float* dev_mu_R;
    UidType* dev_uid_coeffs;
    float* dev_dual_hash;
    half* dev_B;
    float* dev_B_float;
    float* dev_q_transform;
};

inline int n_to_vecdim( const uint32_t n ) {
    return ((n+31)/32) * 32;
}

struct cidx_output;
class GPUStreamGeneral {
    private:
        uint32_t send_b_id_start;
        uint32_t send_b_id_end;
        size_t last_bucketsize;
        size_t last_lift_bucketsize;
        size_t minimum_bucketsize;
        size_t n; 
        const int device;
        const int VECDIM;
        bool lift;
        const size_t dual_hash_vecs;
        const std::string bucketer;
        const bool triple;

        uint32_t bdgl_seed;
        size_t bdgl_blocks;
        size_t bdgl_local_buckets;

        Entry* db;
        CompressedEntry* cdb;
        int const* buckets;
        size_t const* buckets_index;

        const int global;
        const size_t multi_bucket;
        const size_t max_nr_buckets;
        size_t nr_buckets;
        std::vector<size_t> b_cdb_indices;

        std::pair<size_t, size_t> cdb_range;
        std::pair<size_t, size_t> cdb_range_prev;

        triple_bucket const* curr_bucket = nullptr;
        triple_bucket const* prev_bucket = nullptr;
        
        // benchmarking
        const bool benchmark = DEBUG_BENCHMARK;
        bool bench_alternation = 0;
        double bench_sieve_kernel = 0.;
        double bench_sieve_flop = 0.;
        cudaEvent_t start[2];
        cudaEvent_t stop[2];

        // Pinned host memory
        Xtype* host_X; // input
        Xtype* host_X_extend; // output
        lentype* host_len_in; // input
        lentype* host_len_out; // output
        lentype* host_lift_len_out;
        iptype* host_ips; // input & output
        indextype* host_indices; // output
        indextype* host_lift_indices;
        indextype* host_nr_results; // output

        Xtype* dev_X;
        half* dev_X_half;
        float* dev_X_float;
        half* dev_YR_half;
        float* dev_YR_float;
        dhtype* dev_DH;
        Xtype* dev_X_extend;
        lentype* dev_len_in;
        lentype* dev_len_out;
        lentype* dev_lift_len_out;
        half* dev_len_half;
        iptype* dev_ips;
        indextype* dev_indices;
        indextype* dev_lift_indices;
        indextype* dev_nr_results;
    
        // shared global
        half* dev_B;
        float* dev_B_float;
        float* dev_mu_L;
        float* dev_mu_R;
        UidType* dev_uid_coeffs;
        float* dev_dual_hash;
        float* dev_q_transform;

        cudaStream_t stream;
        cublasHandle_t handle;
        cudaEvent_t H2D;
        cudaEvent_t D2H;
    public:
        GPUStreamGeneral(const int device, const size_t n, const std::string gpu_bucketer, const bool gpu_triple, const size_t multi_bucket, const size_t max_nr_buckets, const bool global, std::vector<Entry> &db, std::vector<CompressedEntry> &cdb, const size_t dual_hash_vecs=64);
        void malloc( global_dev_ptrs &dev_ptrs );
        void free();
        void bind(std::vector<Entry> & _db, std::vector<CompressedEntry> & _cdb);

        // general
        // init constant, run once per device per context change
        void reset_results();
        void global_init(const std::vector<std::vector<float>> &mu_L, const std::vector<UidType> &uid_coeffs);        
        void dh_init(const std::vector<std::vector<float>> &mu_L, const std::vector<std::vector<float>> &dual_hash_vectors);

        // helper
        template<bool send_length, bool cdb_index=true>
        void send_X_range( const size_t cdb_start, const size_t cdb_end );
        template <bool send_length, bool cdb_index, bool event_record, class Iterator>
        void send_X_range( const Iterator first, const Iterator last ); 

        inline void X_to_Xhalf( const size_t bucketsize );
        inline void X_to_Xhalf_negate( const size_t bucketsize );
        inline void X_to_Xhalf_normalize( const size_t bucketsize );
        inline void X_to_Xfloat( const size_t bucketsize );
        inline void X_to_Xfloat_negate( const size_t bucketsize );
        inline void X_to_Xfloat_normalize( const size_t bucketsize );
        inline void float_to_half( const float* dev_in, half* dev_out, const size_t bucketsize ); 
        inline void prepare_len_and_ips( float lenbound, float b_len, const size_t bucketsize );
        inline void reorder( half* dev, size_t nr_vecs ); 

        inline void disable_lift() {
            lift = false;
        };
        inline void enable_lift() {
            lift = dual_hash_vecs > 0;
        }

        // bucketing
        void B_init( const std::vector<size_t>& bucket_centers_idx, uint32_t bucket_seed, const std::vector<double> &q, const int force_bdgl_blocks=-1);
        void B_send_data( const size_t cdb_start, const size_t cdb_end );
        void B_launch_kernel();
        void B_receive_data( std::vector<triple_bucket> &bucket, const size_t max_vecs_per_bucket, bool onlyprocess  );

        // processing
        void P_send_data( const triple_bucket &bucket, const float lenbound, const float dh_bucket_ratio ); 
        void P_launch_kernel( uint32_t dh_bound=0  );
        void P_receive_data( queues &queue, bool onlyprocess = false);  

        // extend left
        void E_send_data( const size_t cdb_start, const size_t cdb_end );
        void E_launch_kernel( const size_t extend_left );
        void E_receive_data( const size_t extend_left, bool onlyprocess );

        // recompute
        void R_send_data( const size_t cdb_start, const size_t cdb_end );
        void R_launch_kernel();
        void R_receive_data( bool onlyprocess );
        
        void sync();
        void bench_reset() {
            bench_sieve_flop = 0.;
            bench_sieve_kernel = 0.;
        }
        void bench_results() {
            if( benchmark ) {
            double flops = bench_sieve_flop / bench_sieve_kernel * 1000. / std::pow(2.,40.);

            std::cerr << bench_sieve_flop << " " << bench_sieve_kernel << " " << flops << std::endl;
            } 
            else std::cerr << "Benchmarking disabled" << std::endl; 
        }
};

#endif /* HAVE_CUDA */
#endif /* GPU_STREAM_H_ */
