#include <iostream>

#define DEBUG_BENCHMARK 1
#include "../cuda/GPUStreamGeneral.h"

int main(int argc, char **argv) {
    
    size_t bucket_size = 64*1024;
    int mode = 1; // 0 = only sieve kernel, 1 = everything
    if( argc > 1 )
        bucket_size = atoi(argv[1]);

    if( argc > 2 )
        mode = atoi(argv[2]); 

    const int device = 0;
    const size_t n = 160;
    const std::string gpu_bucketer = "bgj1"; 
    const bool gpu_triple = false; 
    const size_t multi_bucket = 1; 
    const size_t max_nr_buckets = 64*1024;
    const bool global = true;
    std::vector<Entry> db;
    std::vector<CompressedEntry> cdb;
    const size_t dual_hash_vecs=0;
    const size_t bench_buckets = 4*1024 * (64*1024)/bucket_size;
    const float scale = 256;

    // mu_R
    std::vector<std::vector<float>> mu_R(n, std::vector<float>(n, 0.));
    for( size_t i = 0; i < n; i++ ) {
        mu_R[i][i] = 1./(std::sqrt(n)*scale);
    }
    std::vector<UidType> uid_coeffs(n,0);

    // create database
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.,scale);

    const size_t db_size = 2*1024*1024;
    db.resize(db_size);
    cdb.resize(db_size);
    double avglen = 0.;
    for( size_t s = 0; s < db_size; s++ ) {
        // sample x by discretized gaussian
        for( int i = 0; i < n; i++ ) {
            db[s].x[i] = (ZT)(distribution(generator)+.5);
        }
        
        double len = 0.;
        for( int i = 0; i < n; i++ ) {
            len += db[s].x[i] * db[s].x[i] * mu_R[i][i] * mu_R[i][i];
        }
        avglen += len;

        db[s].len = len;
        db[s].uid = 0;
        cdb[s].i = s;
        cdb[s].len = len;
    }
    avglen /= db_size;

    std::vector<indextype> all_indices( db_size );
    for( int i = 0; i < db_size; i++ )
        all_indices[i]=i;
    std::vector<iptype> all_ips( db_size, __float2half(0.01) );
    
    // generate buckets
    size_t nr_buckets = db_size / bucket_size;
    std::vector<triple_bucket> buckets(nr_buckets);
    for( int b = 0; b < nr_buckets; b++ ) {
        buckets[b].indices = &(all_indices[b*bucket_size]);
        buckets[b].ips = &(all_ips[b*bucket_size]);
        buckets[b].size = bucket_size;
        buckets[b].b_db_index = b;
        buckets[b].b_local_index = b;
        buckets[b].b_len = db[b].len;
    }

    size_t streams = 8;
    GPUStreamGeneral* gpu[streams];  
    global_dev_ptrs dev_ptrs;
    for( size_t s = 0; s < streams; s++ ) {
        gpu[s] = new GPUStreamGeneral(device, n, gpu_bucketer, gpu_triple, multi_bucket, max_nr_buckets, s==0, db, cdb, dual_hash_vecs);
        gpu[s]->bind( db, cdb );
        gpu[s]->malloc(dev_ptrs);
    }

    queues queue;
    float lenbound = 1.;
    float lift_ratio = 1.;
    
    gpu[0]->global_init( mu_R, uid_coeffs );
    
    /*
    std::vector<double> q;
    const uint32_t b_seed = 0;
    std::vector<size_t> b_indices( nr_buckets );
    for( int b = 0; b < nr_buckets; b++ )
        b_indices[b] = b;
    gpu->B_init( b_indices, b_seed, q );
    */

    //std::cerr << "Warmup" << std::endl;
     
    thread_pool::thread_pool threadpool;
    threadpool.resize(streams-1);

    std::vector<queues> t_queues(streams);
    

    threadpool.run([nr_buckets, bucket_size, lift_ratio, lenbound, &buckets, &t_queues, &gpu, &db, &cdb](int t_id, int streams)
    {
            gpu[t_id]->bind( db, cdb );        

            for( int b = 0; b < (64*1024/bucket_size)*1024; b++ ) {
                gpu[t_id]->P_send_data( buckets[b%nr_buckets], lenbound, lift_ratio );
                gpu[t_id]->P_launch_kernel( 0 );
                gpu[t_id]->P_receive_data( t_queues[t_id], false );
            }
    
            gpu[t_id]->sync();
            t_queues[t_id].sieve_pairs.clear();
       
    }, streams);

    //std::cerr << "Start run " << std::endl;
    for( int s = 0; s < streams; s++ ) {
        gpu[s]->bench_reset();
        gpu[s]->P_send_data( buckets[s], lenbound, lift_ratio );
    }

       
    auto start = std::chrono::steady_clock::now();
    threadpool.run([mode, bench_buckets, nr_buckets, lift_ratio, lenbound, &buckets, &t_queues, &gpu, &db, &cdb](int t_id, int streams)
    {
            gpu[t_id]->bind( db, cdb );        

            for( int b = 0; b < bench_buckets; b++ ) {
                if( mode > 0 ) gpu[t_id]->P_send_data( buckets[b%nr_buckets], lenbound, lift_ratio );
                gpu[t_id]->P_launch_kernel( 0 );
                if( mode > 0 ) gpu[t_id]->P_receive_data( t_queues[t_id], false );
            }
    
            gpu[t_id]->sync();
            //std::cerr << t_id << " done" << std::endl;       
    }, streams);
    auto stop = std::chrono::steady_clock::now();
    double time_total = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    double flop_total = streams * bench_buckets * n * double(bucket_size) * double(bucket_size);
    
    std::cerr << mode << " " << bucket_size << " " << (flop_total / time_total * 1000. / std::pow(10., 12)) << std::endl;

    for( int s = 0; s < streams; s++ ) {
        //gpu[s]->P_receive_data( queue, true );
        //gpu[s]->bench_results();
        gpu[s]->free();
        delete gpu[s];
    }

    return 0;
}
