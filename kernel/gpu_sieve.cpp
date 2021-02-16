#include <unistd.h>
#include <iomanip>
#include "../cuda/GPUStreamGeneral.h"
#include <immintrin.h>
#include <string.h>


size_t automaxbuckets(size_t max_nr_buckets, size_t dbsize)
{
    if (max_nr_buckets == 0)
    {
        max_nr_buckets = std::max<size_t>(65536, sqrt( double(dbsize)*2 ) );
//        if (max_nr_buckets < params.threads*2)
//            max_nr_buckets = params.threads*2;
//        size_t x = 1;
//        while (x < max_nr_buckets || x < 65536)
//            x *= 2;
        max_nr_buckets = ((max_nr_buckets + 1023)/1024) * 1024;
        std::cerr << "max_nr_buckets = " << max_nr_buckets << std::endl;
    }
    return max_nr_buckets;
}

// -------------------------------- PREPARE GLOBAL ----------------------- //

void Siever::global_init_gpus() {
    
    // mu_R
    std::vector<std::vector<float>> mu_R(n, std::vector<float>(n, 0.));
    for( size_t i = 0; i < n; i++ ) {
        for( size_t j = 0; j < n; j++ ) {
            mu_R[i][j] = muT[i][j] * sqrt_rr[i];
        }
    }

    const auto devices = std::min(params.gpus, gpu_general.size());
    threadpool.run([this, &mu_R](int device)
        {
            gpu_general[device][0]->bind( db, cdb );
            gpu_general[device][0]->global_init( mu_R, uid_hash_table.get_uid_coeffs() );
        }, devices);
}

void Siever::dh_init_gpus() {

    auto nr_vecs = dual_hashes.get_nr_vecs();
    auto dim_vecs = dual_hashes.get_dim_vecs();
    const auto devices = std::min(params.gpus, gpu_general.size());
    for( size_t i = 0; i < gpu_general.size(); i++ ) {
        for( size_t j = 0; j < gpu_general[i].size(); j++ ) {
            if( nr_vecs > 0 )
                gpu_general[i][j]->enable_lift();
            else
                gpu_general[i][j]->disable_lift();
        }
    }

    if( nr_vecs == 0 )
        return;

    // mu_L
    std::vector<std::vector<float>> mu_L(dim_vecs, std::vector<float>(n, 0.));
    std::vector<std::vector<float>> dual_vecs(nr_vecs, std::vector<float>(dim_vecs));
         
    // fill lower part
    for( size_t i = 0; i < dim_vecs; i++ ) {
        for( size_t j = 0; j < n; j++ ) {
            mu_L[i][j] = full_muT[l-i-1][l+j];
        }
    }

    for( size_t i = 0; i < nr_vecs; i++ ) {
        for( size_t j = 0; j < dim_vecs; j++ ) {
            dual_vecs[i][j] = dual_hashes.get_dual_vecs(i, j); 
        }
    }
    threadpool.run([this, &mu_L, &dual_vecs](int device)
        {
            gpu_general[device][0]->bind( db, cdb );
            gpu_general[device][0]->dh_init( mu_L, dual_vecs );
        }, devices);
}


// -------------------------------- BUCKETING ---------------------------- //

void random_orthogonal (std::vector<double> &qspace, int n, int64_t seed) {
    double vnorm, vTa, vpartdot;

	std::vector<double*> q(n);
	for( int i = 0; i < n; i++ )
		q[i] = &qspace[i*n];

    double ** v = new double*[n];
    for(int i = 0; i < n; i++) {
        v[i] = new double[n - i];
    }

    std::default_random_engine generator( seed );
    std::normal_distribution<double> distribution(0.,1.0);

    for( int i = 0; i < n; i++ ) {
    	for( int j = 0; j < n; j++ ) {
    		q[i][j] = distribution(generator);
    	}
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n-i; j++) {
	        v[i][j] = q[i][i + j];
	    }

	    vpartdot = 0.;
	    for( int j = 1; j < n-i; j++ )
	    	vpartdot += v[i][j] * v[i][j];

        if(v[i][0] < 0) {
            v[i][0] -= sqrt(v[i][0] * v[i][0] + vpartdot);
        }
        else {
            v[i][0] += sqrt(v[i][0] * v[i][0] + vpartdot);
        }

        vnorm = 1./sqrt(v[i][0] * v[i][0] + vpartdot);
        for( int j = 0; j < n-i; j++ )
        	v[i][j] *= vnorm;
    
        for(int j = i; j < n; j++) {
            vTa = 0.;
            for(int k = 0; k < n-i; k++ )
            	vTa += q[j][i+k] * v[i][k];
            vTa *= 2;
            for(int k = 0; k < n-i; k++)
            	q[j][i+k] -= vTa * v[i][k];
        }
    }

    std::vector<double> sgns(n,0);
    for( int i = 0; i < n; i++ )
    	sgns[i] = (q[i][i]<0) ? -1. : 1.;

    for( int i = 0; i < n; i++ ) {
    	for( int j = 0; j < n; j++ ) {
    		q[i][j] = double((i==j));
    	}
    }

    std::vector<double> tmp(n, 0.);
    for( int l = n-1; l >= 0; l-- ) {
    	for(int i = l; i < n; i++ ) {
    		tmp[i] = 0.;
    		for( int j = l; j < n; j++ ) {
    			tmp[i] += 2. * q[i][j] * v[l][j-l];
    		}
    	}
    	for( int i = l; i < n; i++ ) {
    		for( int j = l; j < n; j++ ) {
    			q[i][j] -= tmp[i] * v[l][j-l];
    		}
    	}
    }

    // adjust Q based on sign of diag(R), 
    // makes distribution uniform over Haar measure.
    for( int i = 0; i < n; i++ ) {
    	for( int j = 0; j < n; j++ ) {
    		q[i][j] *= sgns[i];
    	}
    }

    for(int i = 0; i < n; i++) {
        delete[] v[i];
    }
    delete[] v;
}

void Siever::gpu_bucketing_prepare( const size_t threads, const std::vector<size_t>& b_idxs) {
    size_t chunks = b_idxs.size();

    uint32_t bucket_seed = (uint32_t)rng();
	std::vector<double> q;

	if( params.gpu_bucketer == "bdgl" ) {
		q.resize(n*n);
		// construct random orthonormal transform
        std::vector<double> orth(n*n);
        random_orthogonal (orth, n, rng());
	
        std::vector<std::vector<double>> mu_R(n, std::vector<double>(n, 0.));
        for( size_t i = 0; i < n; i++ ) {
            for( size_t j = 0; j < n; j++ ) {
                mu_R[i][j] = muT[i][j] * sqrt_rr[i];
            }
        }

        for( int i = 0; i < n; i++ ) {
            for( int j = 0; j < n; j++ ) {
                double tmp = 0.;
                for( int k = 0; k < n; k++ ) {
                    tmp += orth[n*i+k] * mu_R[k][j];
                }
                q[n*i+j] = tmp;
            }
        }
    }
    
    threadpool.run([this, chunks, &b_idxs, &q, bucket_seed](int th_i, int th_n)
        {
            size_t streams = std::min( gpu_general[0].size(), chunks/th_n + (th_i < (chunks%th_n)));
            for( size_t s = 0; s < streams; ++s)
            {
                gpu_general[th_i][s]->bind( db, cdb);
                gpu_general[th_i][s]->B_init( b_idxs, bucket_seed, q );
            }
        }, threads);
}
 
void Siever::gpu_bucketing_process_task( const size_t A, const size_t t_id, const size_t threads, const size_t chunk_size, const size_t vecs_per_local_part, std::vector<triple_bucket> &t_buckets) {
    size_t chunks = (A+chunk_size-1) / chunk_size;
    size_t streams = std::min( gpu_general[0].size(), chunks/threads + (t_id < (chunks%threads )));
    
    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->bind(db, cdb);
    }

    // TODO: divide work dynamically using atomics?
    for( size_t cdb_start = t_id * chunk_size, s = 0; cdb_start < A; cdb_start += threads * chunk_size, s++ ) {
        size_t cdb_end = std::min( cdb_start+chunk_size, A );

        gpu_general[t_id][s%streams]->B_send_data( cdb_start, cdb_end );
        gpu_general[t_id][s%streams]->B_launch_kernel();
        gpu_general[t_id][s%streams]->B_receive_data( t_buckets, vecs_per_local_part, false );
    }

    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->B_receive_data( t_buckets, vecs_per_local_part, true );
    }

}


void Siever::gpu_bucketing( const size_t A, size_t chunk_size, const std::vector<size_t>& bucketcenters, size_t threads, std::vector<triple_bucket> &buckets, std::vector<indextype> &all_bucket_indices, std::vector<iptype> &all_bucket_ips ) {
    
    size_t nr_buckets = bucketcenters.size();
    buckets.resize(nr_buckets);
    
    // special case if only 1 bucket
    if( nr_buckets == 1 ) {
        buckets[0].indices = all_bucket_indices.data();
        buckets[0].ips = all_bucket_ips.data();
        
        size_t b_start2 = bucketcenters[0];
        
        for( size_t i = 0; i < b_start2; i++ )
            buckets[0].indices[i] = cdb[i].i;
        for( size_t i = b_start2+1; i < A; i++ )
            buckets[0].indices[i-1] = cdb[i].i;
        buckets[0].size = A-1;

        // random directions
        half dirs[2] = {__float2half(0.01), __float2half(-0.01)};
        for( size_t i = 0; i < A-1; i++ )
            buckets[0].ips[i] = dirs[rng()%2];

        buckets[0].b_db_index = b_start2;
        // too large value to prevent triples (saves computing the ips)
        buckets[0].b_len = 100.;
        return;
    }

    auto start_time = std::chrono::steady_clock::now();
    

    threads = std::min( threads, (A+chunk_size-1) / chunk_size); 

    // Prepare bucket vectors
    gpu_bucketing_prepare( threads, bucketcenters );
    benchB_prepare += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
    start_time = std::chrono::steady_clock::now();

    // partition local views
    std::vector<std::vector<triple_bucket>> t_buckets(threads, std::vector<triple_bucket>(nr_buckets));
    const size_t vecs_per_bucket = all_bucket_indices.size() / nr_buckets;
    const size_t vecs_per_local_part = vecs_per_bucket / threads;
    for( size_t b = 0; b < nr_buckets; b++ ) {
        buckets[b].indices = all_bucket_indices.data() + b * vecs_per_bucket;
        buckets[b].ips = all_bucket_ips.data() + b * vecs_per_bucket;
        for( size_t t_id = 0; t_id < threads; t_id++ ) {
            t_buckets[t_id][b].indices = buckets[b].indices + t_id * vecs_per_local_part;
            t_buckets[t_id][b].ips = buckets[b].ips + t_id * vecs_per_local_part;
            t_buckets[t_id][b].size = 0;
        }
    }


    // Do the actual bucketing 
    threadpool.run([this, A, &t_buckets, chunk_size, vecs_per_local_part]
        (int t_id, int threads)
        {
            gpu_bucketing_process_task(A, t_id, threads, chunk_size, vecs_per_local_part, t_buckets[t_id] );
        }, threads);

    benchB_bucketing += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
    start_time = std::chrono::steady_clock::now();


    // Merge results
    threadpool.run([this, nr_buckets, &buckets, &t_buckets, &bucketcenters, vecs_per_local_part](int t_id, int threads)
        {
            for( size_t b = t_id; b < nr_buckets; b+=threads ) {
                size_t bsize = 0;
                for( size_t i = 0; i < threads; i++ ) {
                    t_buckets[i][b].size = std::min( t_buckets[i][b].size, uint32_t(vecs_per_local_part) );    
                    bsize += t_buckets[i][b].size;
                }

                buckets[b].size = bsize;
                buckets[b].b_db_index = cdb[bucketcenters[b]].i;
                buckets[b].b_local_index = b;
                buckets[b].b_len = cdb[bucketcenters[b]].len;
                
                size_t index = 0;
                for( size_t i = 0; i < threads; i++ ) {
                    std::copy( t_buckets[i][b].indices, t_buckets[i][b].indices + t_buckets[i][b].size, buckets[b].indices+index );
                    std::copy( t_buckets[i][b].ips, t_buckets[i][b].ips + t_buckets[i][b].size, buckets[b].ips+index );
                    index += t_buckets[i][b].size;    
                }
            }
        }, threads);
    benchB_merge += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
}

// -------------------------------- PROCESSING ------------------------------------ //

void Siever::gpu_processing_task( const size_t t_id, const size_t threads, const float lenbound, const std::vector<triple_bucket> &buckets, queues &t_queue, size_t max_results ) {
    const size_t nr_buckets = buckets.size();
    const size_t streams = std::min( gpu_general[0].size(), nr_buckets/threads + (t_id < (nr_buckets%threads )));

    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->bind(db, cdb);
    }

    // TODO: divide work dynamically using atomics?
    size_t b = t_id;
    std::vector<bool> did_work(streams, false);
    for( size_t s = 0 ;b < nr_buckets; b += threads, s++  ) {
        if( buckets[b].size < 128 )
            continue;
        did_work[s%streams] = true;
        gpu_general[t_id][s%streams]->P_send_data( buckets[b], lenbound, params.dh_bucket_ratio );
        gpu_general[t_id][s%streams]->P_launch_kernel( dual_hashes.get_dh_bound() );

        size_t queue_pairs = t_queue.sieve_pairs.size();
        size_t queue_triples = t_queue.sieve_triples.size();
        gpu_general[t_id][s%streams]->P_receive_data( t_queue, false );
        for (size_t i = queue_pairs; i < t_queue.sieve_pairs.size(); ++i)
        {
            UidType new_uid = compute_uid<2>( t_queue.sieve_pairs[i] );
            if(! uid_hash_table.check_uid_unsafe(new_uid) )
                continue;
            std::swap(t_queue.sieve_pairs[i], t_queue.sieve_pairs.back());
            t_queue.sieve_pairs.pop_back();
            --i;
        }
        for (size_t i = queue_triples; i < t_queue.sieve_triples.size(); ++i)
        {
            UidType new_uid = compute_uid<3>( t_queue.sieve_triples[i] );
            if(! uid_hash_table.check_uid_unsafe(new_uid) )
                continue;
            std::swap(t_queue.sieve_triples[i], t_queue.sieve_triples.back());
            t_queue.sieve_triples.pop_back();
            --i;
        }

        if (params.otf_lift)
            gpu_sieve_lift( t_queue );
        if (t_queue.sieve_pairs.size() + t_queue.sieve_triples.size() > max_results)
            break;
    }

    for( size_t i = 0; i < streams; i++ ) {
        if( did_work[i] )
            gpu_general[t_id][i]->P_receive_data( t_queue, true );
    }
    if (params.otf_lift)
	gpu_sieve_lift( t_queue ); 
}

void Siever::gpu_processing( const size_t threads, const float lenbound, const std::vector<triple_bucket> &buckets, std::vector<queues> &t_queue, size_t max_results) 
{
    threadpool.run([this, lenbound, &buckets, &t_queue, max_results](int t_id, int threads)
        {
            gpu_processing_task( t_id, threads, lenbound, buckets, t_queue[t_id], max_results / threads);
        }, threads);
}

// -------------------------------- RECOMPUTE ------------------------------------ //
void Siever::gpu_recompute_task( const size_t A, const size_t t_id, const size_t threads, const size_t chunk_size) {
    size_t chunks = (A+chunk_size-1) / chunk_size;
    size_t streams = std::min( gpu_general[0].size(), chunks/threads + (t_id < (chunks%threads )));

    if( streams == 0 )
        return;

    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->bind(db, cdb);
    }

    for( size_t db_start = t_id * chunk_size, s = 0; db_start < A; db_start += threads * chunk_size, s++ ) {
        size_t db_end = std::min( db_start+chunk_size, A );

        gpu_general[t_id][s%streams]->R_send_data( db_start, db_end );
        gpu_general[t_id][s%streams]->R_launch_kernel();
        gpu_general[t_id][s%streams]->R_receive_data( false );
    }

    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->R_receive_data( true );
    }
}

void Siever::gpu_recompute( const size_t A, const size_t threads ) {   
    size_t remainder = A%32;

    size_t chunk_size = 64*1024;
    threadpool.run([this, A, remainder, chunk_size](int t_id, int threads)
        {
            gpu_recompute_task(A-remainder, t_id, threads, chunk_size );
        }, threads);

    for( size_t i = A-remainder; i < A; i++ ) {
        recompute_data_for_entry<Recompute::recompute_yr | Recompute::recompute_len | Recompute::recompute_uid >(db[i]);
    }
}

void Siever::gpu_extend_task( const size_t A, const size_t extend_left, const size_t t_id, const size_t threads, const size_t chunk_size) {
    size_t chunks = (A+chunk_size-1) / chunk_size;
    size_t streams = std::min( gpu_general[0].size(), chunks/threads + (t_id < (chunks%threads )));

    if( streams == 0 )
        return;

    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->bind(db, cdb);
    }

    for( size_t db_start = t_id * chunk_size, s = 0; db_start < A; db_start += threads * chunk_size, s++ ) {
        size_t db_end = std::min( db_start+chunk_size, A );
        gpu_general[t_id][s%streams]->E_send_data( db_start, db_end );
        gpu_general[t_id][s%streams]->E_launch_kernel(extend_left);
        gpu_general[t_id][s%streams]->E_receive_data( extend_left, false );
    }

    for( size_t i = 0; i < streams; i++ ) {
        gpu_general[t_id][i]->E_receive_data( extend_left, true );
    }
}

void Siever::gpu_extend( const size_t A, const size_t extend_left, const size_t threads ) {
    size_t remainder = A%32;

    size_t chunk_size = 64*1024;
    threadpool.run([this, A, remainder, extend_left, chunk_size](int t_id, int threads)
        {
            gpu_extend_task(A-remainder, extend_left, t_id, threads, chunk_size );
        }, threads);

    for( size_t i = A-remainder; i < A; i++ ) {
        recompute_data_for_entry_babai<Recompute::babai_only_needed_coos_and_recompute_aggregates | Recompute::recompute_yr>(db[i],extend_left);
    }
}

// ------------------------------ QUEUE INSERT ---------------------------- //

template<size_t tuple_size>
inline UidType Siever::compute_uid( const Qtuple<tuple_size> &q ) {
    UidType new_uid = db[q.v[0]].uid;
    for( size_t i = 0; i < tuple_size-1; i++ ) {
        if( (q.sign)&(1<<i) )
            new_uid -= db[q.v[i+1]].uid;
        else
            new_uid += db[q.v[i+1]].uid;
    }
    return new_uid;
}

template<size_t tuple_size>
inline std::array<ZT,MAX_SIEVING_DIM> Siever::compute_x( const Qtuple<tuple_size> &q) {
    auto new_x = db[q.v[0]].x;
    for( size_t i = 0; i < tuple_size-1; i++ ) {
        addsub_vec(new_x,  db[q.v[i+1]].x, static_cast<ZT>( ((q.sign)&(1<<i))?-1:1 ));
    }
    return new_x;
}

void Siever::gpu_sieve_lift( queues &queue) {
    // read only duplicate removal, doesn't need locks on hash table

    // pairs
    {
        const size_t Q = queue.lift_pairs.size();
        for( size_t i = 0; i < Q; i++ ) {
            ZT x[r];
            std::fill (x,x+l,0);
            
            ZT* x1 = db[queue.lift_pairs[i].v[0]].x.data();
            ZT* x2 = db[queue.lift_pairs[i].v[1]].x.data();
            if( queue.lift_pairs[i].sign ) {
                for( unsigned int j = 0; j < n; j++ )
                    x[j+l] = x1[j] - x2[j];
            } else {
                for (unsigned int j = 0; j < n; ++j)
                    x[j+l] = x1[j] + x2[j];
            }
            lift_and_compare(x, queue.lift_pairs[i].len * gh, nullptr );
        }
        queue.lifted_pairs += Q;
		queue.lift_pairs.clear();
		queue.lift_triples.clear();
    }
    
}


void Siever::gpu_sieve_duplicate_remove( queues &queue, size_t max_results ) 
{
    // read only duplicate removal, doesn't need locks on hash table
    // if already present, use negative len as duplicate marker
    size_t count = 0;
    // triples
    {
        for( size_t i = 0; i < queue.sieve_triples.size(); ++i) {
            UidType new_uid = compute_uid<3>( queue.sieve_triples[i] );
            if( uid_hash_table.check_uid_unsafe(new_uid) )
                queue.sieve_triples[i].len = -1;
            else
                if (++count == max_results)
                {
                    queue.sieve_triples.resize(i+1);
                    queue.sieve_pairs.clear();
                    return;
                }
        }
    }

    // pairs
    {
        for( size_t i = 0; i < queue.sieve_pairs.size(); ++i) {
            UidType new_uid = compute_uid<2>( queue.sieve_pairs[i] );
            if( uid_hash_table.check_uid_unsafe(new_uid) )
                queue.sieve_pairs[i].len = -1;
            else
                if (++count == max_results)
                {
                    queue.sieve_pairs.resize(i+1);
                    return;
                }
        }
    }
    
}

template<size_t tuple_size>
void Siever::gpu_sieve_delayed_replace( const Qtuple<tuple_size> &q, std::deque<Entry> &transaction_db ) {
    Entry new_entry;
    new_entry.x = compute_x<tuple_size>( q ); 
    UidType new_uid = uid_hash_table.compute_uid(new_entry.x);
    if( uid_hash_table.insert_uid(new_uid) ) {
        new_entry.uid = new_uid;
        new_entry.len = q.len;
        recompute_data_for_entry<Recompute::recompute_len | Recompute::consider_otf_lift | Recompute::recompute_otf_helper>(new_entry); 
        transaction_db.push_back( new_entry );
    }
}

void Siever::gpu_sieve_queue_to_entry( queues &queue, std::deque<Entry> &transaction_db, const size_t max_results ) 
{
    // triples
    {
        while (!queue.sieve_triples.empty())
        {
            if( queue.sieve_triples.back().len >= 0. )
            {
                gpu_sieve_delayed_replace<3>( queue.sieve_triples.back(), transaction_db );
            }
            queue.sieve_triples.pop_back();
            if( transaction_db.size() >= max_results )
                return;
        }
    }


    // pairs
    {
        while (!queue.sieve_pairs.empty())
        {
            if( queue.sieve_pairs.back().len >= 0. )
            {
                gpu_sieve_delayed_replace<2>( queue.sieve_pairs.back(), transaction_db );
            }
            queue.sieve_pairs.pop_back();
            if( transaction_db.size() >= max_results )
                return;
        }
    }
}

bool Siever::gpu_sieve_replace_in_db(size_t cdb_index, const Entry &e){
    CompressedEntry &ce = cdb[cdb_index];
    if (REDUCE_LEN_MARGIN_HALF * e.len >= ce.len)
    {
        uid_hash_table.erase_uid(e.uid);
        return false;
    }
    uid_hash_table.erase_uid(db[ce.i].uid);
    ce.bucket_center_count = 0;
    ce.len = e.len;
    db[ce.i] = e;
    return true;
}

void Siever::gpu_sieve_replace( const size_t t_id, const size_t threads, std::deque<Entry> &transaction_db, size_t &min_i_index ) {
    int64_t i_index = min_i_index;
    int64_t t_index = 0;
    for(; !transaction_db.empty() && i_index >= threads; ++t_index)
    {
        if( gpu_sieve_replace_in_db( i_index, transaction_db.front() ))
            i_index -= threads;
        transaction_db.pop_front();
    }
    min_i_index = size_t(i_index);    
    transaction_db.clear();
}

void Siever::gpu_insert_queue( const size_t threads, std::vector<queues> &t_queue, size_t max_results ) {
    std::vector<std::deque<Entry>> transaction_db(threads);
    std::vector<size_t> inserted_count(threads, 0);
    std::vector<size_t> min_i_index(threads, db.size());

    threadpool.run([this, &t_queue, &transaction_db, &inserted_count, &min_i_index, max_results](int t_id, int threads)
        {
            min_i_index[t_id] = cdb.size() - 1 - t_id;
            while (inserted_count[t_id] < (max_results / threads))
            {
                gpu_sieve_queue_to_entry( t_queue[t_id], transaction_db[t_id], 4096); //max_results/threads);
                if (transaction_db[t_id].empty())
                    break;
                    
                gpu_sieve_replace( t_id, threads, transaction_db[t_id], min_i_index[t_id]);
                
                transaction_db[t_id].clear();

                inserted_count[t_id] = ( (cdb.size() - 1 - t_id) - min_i_index[t_id] ) / threads;
            }
            t_queue[t_id].sieve_pairs.clear();
            t_queue[t_id].sieve_triples.clear();
        }, threads);

    gpu_stats_nonduplicates = 0;
    for (auto& tdbi : transaction_db)
        gpu_stats_nonduplicates += tdbi.size();
    gpu_stats_inserted = 0;
    for (auto ic : inserted_count)
        gpu_stats_inserted += ic;

    status_data.plain_data.sorted_until = *min_element(min_i_index.begin(), min_i_index.end());
}

// ------------------------------ GPU management ------------------------ //
void Siever::gpu_general_smartInit() {
   
    int new_VECDIM = ((n+31)/32)*32;
    const auto threads = params.threads;
    const auto devices = params.gpus;
    const auto dh_vecs = params.dh_vecs;
    const size_t streams_per_thread = params.streams_per_thread;
    const size_t multi_bucket = params.multi_bucket;
    const size_t max_nr_buckets = automaxbuckets(params.max_nr_buckets, db.capacity());
    const std::string gpu_bucketer = params.gpu_bucketer;
    const bool gpu_triple = params.gpu_triple;

    assert( gpu_bucketer == "bgj1" or gpu_bucketer == "bdgl" );
    assert( gpu_bucketer != "bdgl" or gpu_triple == false );

    if( (not gpu_general_initialized) or (gpu_general_VECDIM != new_VECDIM) ) {
        
        if( gpu_general_initialized )
            gpu_general_freeAll();
 
        gpu_general.resize(threads);
        for( size_t i = 0; i < devices; i++ ) {
            threadpool.push([this, i, devices, threads, streams_per_thread, gpu_bucketer, gpu_triple, multi_bucket, max_nr_buckets, dh_vecs]() {
                global_dev_ptrs dev_ptrs;
                for( size_t j = i; j < threads; j+=devices ) {
                    gpu_general[j].resize(streams_per_thread);
                    for( size_t s = 0; s < streams_per_thread; s++ ) {
                        gpu_general[j][s] = new GPUStreamGeneral( i, n, gpu_bucketer, gpu_triple, multi_bucket, max_nr_buckets, (j==i) and (s==0), this->db, this->cdb, dh_vecs);
                        gpu_general[j][s]->malloc(dev_ptrs);
                    }
                }
            });
        }
        threadpool.wait_work();
        gpu_general_VECDIM = new_VECDIM;
        gpu_general_initialized = true;
    } 
    assert( gpu_general.size() == params.threads );
    assert( gpu_general[0].size() == streams_per_thread );
}

void Siever::gpu_general_freeAll() {
    // free all
    size_t threads = gpu_general.size();
    if( threads > 0 ) {
        size_t streams_per_thread = gpu_general[0].size();

        for( size_t t_id = 0; t_id < threads; t_id++ ) {
            threadpool.push([this, t_id, streams_per_thread]() {
                for( size_t s = 0; s < streams_per_thread; s++ ) {
                    gpu_general[t_id][s]->bind( this->db, this->cdb );
                    gpu_general[t_id][s]->free();
                    delete gpu_general[t_id][s];
                    gpu_general[t_id][s] = nullptr;
                }
                gpu_general[t_id].clear();
            });
        }
        threadpool.wait_work();
        gpu_general.clear();
    }
    gpu_general_initialized = false;
}

// ------------------------------ TRIPLE SIEVE GPU ---------------------- //
void Siever::gpu_sieve() {
    //std::cerr << "Start triple n = " << n << std::endl;
    // round statistics
    std::vector<double> time_lenbound;
    std::vector<double> time_bucketing;
    std::vector<double> time_processing;
    std::vector<double> flops_sieve_processing;
    std::vector<double> time_queue;
    std::vector<double> time_total;
    std::vector<double> hist_lenbound;
    std::vector<size_t> hist_nr_buckets;
    std::vector<size_t> hist_A;
    std::vector<size_t> hist_bsize;
    std::vector<double> hist_bsize_var;
    std::vector<size_t> hist_queue_pairs_size;
    std::vector<size_t> hist_queue_triples_size;
    std::vector<size_t> hist_queue_pairs_lift;

    std::vector<size_t> hist_wrong_length;
    std::vector<size_t> hist_duplicate;
    std::vector<size_t> hist_nonduplicates;
    std::vector<size_t> hist_inserted;
    std::vector<double> hist_acceptance_ratio;

    size_t threads = params.threads;
    const size_t multi_bucket = params.multi_bucket;
    const size_t max_nr_buckets = automaxbuckets(params.max_nr_buckets, db.size());
    double lenbound_ratio = params.lenbound_ratio;
    size_t A = size_t(0.5 * db.size());
    A -= (A%256);
    assert( cdb.size() == uid_hash_table.hash_table_size() );
    switch_mode_to(SieveStatus::plain);
    parallel_sort_cdb();
    recompute_histo();
    for (auto& ce : cdb)
        ce.bucket_center_count = 0;
        
    double max_lenbound = cdb[.99 * std::sqrt(3./4.) * db.size()].len * 1.005;
    if (n >= 80)
        max_lenbound = std::min<double>(max_lenbound, 1.125 * params.saturation_radius);
    const size_t max_peak_results = double(cdb.size()) * std::max<double>(.1, std::min<double>(.3, .3 - .01*(float(n)-110.) ));
    
    size_t saturation_index = 0.5 * params.saturation_ratio * std::min<size_t>(db.size(), std::pow(params.saturation_radius, n/2.0));
    if( saturation_index > 0.5 * A ) {
        std::cerr << "Saturation index larger than half of db size" << std::endl;
        saturation_index = std::min(saturation_index, A-1);
    }

    std::vector<triple_bucket> buckets;

    size_t chunk_size = 64*1024;
    double bucket_overflow_f = 1.2;
    size_t bucket_space = size_t(bucket_overflow_f * (multi_bucket * db.size() + threads * chunk_size - (multi_bucket*db.size())%(threads*chunk_size)  ));
    
	std::vector<indextype> all_bucket_indices(bucket_space);
    std::vector<iptype> all_bucket_ips(bucket_space);
    std::vector<size_t> bucketcenters;
    std::mutex mut;
    size_t max_bucket_center_count = 1;

    // compute nr of buckets
    size_t it = 0;
    size_t last_nr_results = db.size();
    bool saturated = false;
    while( !saturated ) {
        auto start_total = std::chrono::steady_clock::now();
        

        // ------------- determine lenbound ------------------ //
        auto start_lenbound = std::chrono::steady_clock::now();
        float lenbound;
        size_t max_results = max_peak_results;
#if 1
        // max_results = (30% for d <= 110, linearly scales to 10% for d >= 130)
        max_results = std::min<double>(double(last_nr_results) * 1.5, max_peak_results);
        lenbound = std::min<double>( cdb[cdb.size() - max_results].len, max_lenbound );
        //std::cerr << "max_results = " << max_results << " lenbound=" << lenbound << " #uid=" << uid_hash_table.hash_table_size() << std::endl;
#else
        {
            if( hist_inserted.size() > 0 and hist_inserted.back() < 0.02 * db.size() ) {
                lenbound = hist_lenbound.back();
                lenbound = std::min(cdb[cdb.size()-1].len, lenbound);
                lenbound_ratio = 1.;
            } else {
                std::vector<double> sum_lengths(params.threads);
                size_t vecs = std::min(cdb.size(), size_t(lenbound_ratio * cdb.size()));
                size_t vecs_per_thread = (vecs+params.threads-1)/params.threads;
                
                //std::cerr << "Gather vecs: " << vecs << std::endl;

                for( size_t t_id = 0; t_id < params.threads; t_id++ ) {
                    threadpool.push([this, t_id, &sum_lengths, vecs, vecs_per_thread]() {
                        size_t i_start = t_id * vecs_per_thread;
                        size_t i_end = std::min( (t_id+1) * vecs_per_thread, vecs);
                        double sm = 0.;
                        for( size_t i = i_start; i < i_end; ++i )
                            sm += cdb[i].len;
                        sum_lengths[t_id] = sm;    
                    });
                }
                threadpool.wait_work();

                //std::cerr << "Done" << std::endl;

                double avg = 0.;
                for( size_t t_id = 0; t_id < params.threads;  t_id++ )
                    avg += sum_lengths[t_id];
                avg /= vecs;
                lenbound = avg;
            }
            lenbound = std::max( lenbound, float(params.saturation_radius));
        }
#endif
        //std::cerr << "Update dh bound" << std::endl;
        dual_hashes.update_dh_bound(*this, lenbound );
        hist_lenbound.push_back(lenbound);
        time_lenbound.push_back( std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_lenbound).count());

        // -------------- determine Sieve part db ------------- //
        // A = { all vecs with size <= f * lenbound )
        float A_factor = 1.1; // TODO: parametrize
        auto Comp = [](CompressedEntry const &ce, double const &bound){return ce.len < bound; };
        A = std::lower_bound(cdb.cbegin(), cdb.cend(), A_factor * lenbound, Comp) - cdb.cbegin();
        A = std::min(A, cdb.size());
        A -= A%256;

        // -------------- determine nr of buckets ------------- //
        size_t nr_buckets = 1;
   
        if( params.gpu_bucketer == "bgj1" ) {
            if( A*multi_bucket > 2e6 ) {
                nr_buckets = (A*multi_bucket)/(32*1024);
                if( nr_buckets > 32*1024 ) {
                    nr_buckets = std::sqrt(double(A)*double(multi_bucket) / double(max_bucket_center_count));
                }
                nr_buckets -= nr_buckets % 32;
            } else {
                nr_buckets = (A*multi_bucket)/(8*1024);
                nr_buckets -= nr_buckets % 32;
                if( nr_buckets == 0 )
                    nr_buckets = 1;
            }
            nr_buckets = std::min(nr_buckets, max_nr_buckets);
        }
        else if (params.gpu_bucketer == "bdgl" ) {
            nr_buckets = A * multi_bucket / params.bdgl_bucket_size;
            if( nr_buckets < 2*multi_bucket or A < 64*1024 )
                nr_buckets = 1;

            // 2-block bdgl, make sure same transitioning point is used in GPUStreamGeneral
            if( nr_buckets > 8*1024 ) {
                // Force nr_buckets= 2*x^2 for some x
                size_t x = (size_t)std::sqrt(nr_buckets/2.);
                nr_buckets = 2 * x * x;
            }
        }

        hist_nr_buckets.push_back(nr_buckets);
        hist_A.push_back(A);

        // adjust threads for small number of buckets
        threads = std::min(params.threads, nr_buckets);

        // ---------------- bucketing -------------- // 
        auto start_bucketing = std::chrono::steady_clock::now();
        if( params.gpu_bucketer == "bgj1" )
        {
            // find unused bucket centers with parallel linear search
            bucketcenters.clear();
            threadpool.run([this,&bucketcenters,&mut,nr_buckets](int th_i, int th_n)
                {
                    std::vector<std::vector<size_t>> tmps;
                    pa::subrange sr(cdb.size(), th_i, th_n);
                    for (auto j : sr)
                    {
                        auto jc = cdb[j].bucket_center_count;
                        if (jc >= tmps.size())
                            tmps.resize( jc + 1 );
                        if (tmps[jc].size() < nr_buckets)
                        {
                            tmps[jc].emplace_back(j);
                            if (jc == 0 && tmps[0].size() == nr_buckets)
                                break;
                        }
                    }
                    for (size_t j = 1; j < tmps.size() && tmps[0].size()<nr_buckets; ++j)
                        for (size_t k = 0; k < tmps[j].size() && tmps[0].size()<nr_buckets; ++k)
                            tmps[0].emplace_back( tmps[j][k] );
                    std::lock_guard<std::mutex> lock(mut);
                    bucketcenters.insert(bucketcenters.end(), tmps[0].begin(), tmps[0].end());
                });
            auto cf = [this](const size_t lhs, const size_t rhs)
                {
                    if (cdb[lhs].bucket_center_count != cdb[rhs].bucket_center_count)
                        return cdb[lhs].bucket_center_count < cdb[rhs].bucket_center_count;
                    return lhs < rhs;
                };
            pa::nth_element(bucketcenters.begin(), bucketcenters.begin()+nr_buckets, bucketcenters.end(), cf, threadpool);
            bucketcenters.resize(nr_buckets);
            for (const auto& i : bucketcenters)
            {
                if (++cdb[i].bucket_center_count > max_bucket_center_count)
                    max_bucket_center_count = cdb[i].bucket_center_count;
            }
        }
        else if( params.gpu_bucketer == "bdgl" ) {
            bucketcenters.resize(nr_buckets);
        }

        gpu_bucketing( A, chunk_size, bucketcenters, threads, buckets, all_bucket_indices, all_bucket_ips ); 
        time_bucketing.push_back( std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_bucketing).count());

        // bucketing statistics
        double total_flops = 0.;
        {
            size_t mn = A;
            size_t mx = 0;
            double avg = 0.;
            double avgsquared = 0.;
            for( size_t i = 0; i < buckets.size(); i++ ) {
                mn = std::min( mn, size_t(buckets[i].size) );
                mx = std::max( mx, size_t(buckets[i].size) );
                avg += buckets[i].size;
                avgsquared += buckets[i].size * buckets[i].size;
                total_flops += double(buckets[i].size) * double(buckets[i].size) * ((n+31)/32)*32;
            }
            avg /= buckets.size();
            avgsquared /= buckets.size();
            hist_bsize.push_back(size_t(avg+0.5));
            hist_bsize_var.push_back( std::sqrt(avgsquared - avg*avg)  );
        }        

        // ---------------- process buckets -------- //
        auto start_processing = std::chrono::steady_clock::now();
        std::vector<queues> t_queue(threads);
        gpu_processing( threads, lenbound, buckets, t_queue, 3 * max_peak_results);
        time_processing.push_back(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_processing).count());

        // count results
        {
            size_t results_pairs = 0;
            size_t results_triples = 0;
            size_t results_pairs_lifts = 0;
            for( size_t i = 0; i < threads; i++ ) {
                results_pairs += t_queue[i].sieve_pairs.size(); 
                results_triples += t_queue[i].sieve_triples.size();
                results_pairs_lifts += t_queue[i].lift_pairs.size() + t_queue[i].lifted_pairs;
            } 
            hist_queue_pairs_size.push_back(results_pairs);
            hist_queue_triples_size.push_back(results_triples);
            hist_queue_pairs_lift.push_back(results_pairs_lifts);
            flops_sieve_processing.push_back( total_flops / time_processing.back() * 1e6 / double(1024.*1024.*1024.*1024.) ); 
        }
        
#if 0
        // verification
        if( false ) {
            size_t bad_pairs = 0;
            size_t bad_triples = 0;
            size_t bad_pairs_len = 0;
            size_t bad_triples_len = 0;

            // check results
            for( int t = 0; t < threads; t++ ) {
                for( int r = 0; r < t_queue[t].sieve_pairs.size(); r++ ) {
                    auto p = t_queue[t].sieve_pairs[r];
                    uint32_t i = p.v[0];
                    uint32_t j = p.v[1];
                    float len = 0.;
                    if( p.sign ) {
                        for( int k = 0; k < n; k++ ) {
                            float tmp = db[i].yr[k] - db[j].yr[k];
                            len += tmp * tmp;
                        }
                    }
                    else {
                        for( int k = 0; k < n; k++ ) {
                            float tmp = db[i].yr[k] + db[j].yr[k];
                            len += tmp * tmp;
                        }
                    }
                    if( len > 1.01*lenbound ) { 
                        bad_pairs++;
                        std::cerr << "Pair " << t << ", " << r << ": " << p.v[0] << " " << p.v[1] << " " << db[i].len << " " << db[j].len << std::endl;
                        std::cerr << "Len: " << len << std::endl;
                    }
                    
                    if( std::abs(t_queue[t].sieve_pairs[r].len - len) > 0.01 ) {
                        bad_pairs_len++;
                        std::cerr << "Wrong gpu length pair: " << t_queue[t].sieve_pairs[r].len << " " << len << std::endl;
                    }


                    t_queue[t].sieve_pairs[r].len = len;
                }

                for( int r = 0; r < t_queue[t].sieve_triples.size(); r++ ) {
                    auto p = t_queue[t].sieve_triples[r];
                    uint32_t b = p.v[0];
                    uint32_t j = p.v[2];
                    uint32_t i = p.v[1];
                    if( b == i or b == j )
                        std::cerr << "Element in own bucket" << std::endl;
                    float len = 0.;
                    float sgn1 = (p.sign&1) ? -1. : 1.;
                    float sgn2 = (p.sign&2) ? -1. : 1.;
                    float ipbi = 0.;
                    float ipbj = 0.;
                    float ipij = 0.;
                    for( int k = 0; k < n; k++ ) {
                        float tmp = db[b].yr[k] + sgn1 * db[i].yr[k] + sgn2 * db[j].yr[k];
                        ipbi += db[b].yr[k] * db[i].yr[k];
                        ipbj += db[b].yr[k] * db[j].yr[k];
                        ipij += db[i].yr[k] * db[j].yr[k];
                        len += tmp * tmp;
                    }
                    if( len > 1.01*lenbound ) { 
                        bad_triples++;
                        std::cerr << "Triple " << t << ", " << r << ": " << p.v[0] << " " << p.v[1] << " " << p.v[2] << " " << db[i].len << " " << db[j].len << " ::: " << ipbi << " " << ipbj << " " << ipij << " " << int(p.sign) << " " << (p.sign&1) << " " << (p.sign&2) << std::endl;
                        std::cerr << "Len: " << len << std::endl;
                    }
                    
                    if( std::abs(t_queue[t].sieve_triples[r].len - len) > 0.01 ) {
                        bad_triples_len++;
                        std::cerr << "Wrong gpu length triples: " << t_queue[t].sieve_triples[r].len << " " << len << std::endl;
                    }


                    t_queue[t].sieve_triples[r].len = len;
                }
                
                if( bad_triples > 0 or bad_pairs > 0 )
                    std::cerr << "Bad vecs: " << bad_pairs << " " << bad_triples << std::endl;
                if( bad_pairs_len > 0 or bad_triples_len > 0 )
                    std::cerr << "Bad lens: " << bad_pairs_len << " " << bad_triples_len << std::endl;
            }
        }
#endif

        // --------------- insert in db ------------ //
        auto start_queue = std::chrono::steady_clock::now();
        gpu_insert_queue( threads, t_queue, max_peak_results );
        hist_nonduplicates.push_back( gpu_stats_nonduplicates );
        last_nr_results = gpu_stats_inserted;
        hist_inserted.push_back( gpu_stats_inserted );
        time_queue.push_back(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_queue).count());

        // --------------- check saturation -------- //
        parallel_sort_cdb();
        if( cdb[saturation_index].len <= params.saturation_radius
            || (params.goal_r0 > 0 && best_lifts_so_far[ll].len > 0 && best_lifts_so_far[ll].len <= params.goal_r0)
            ) {
            assert(std::is_sorted(cdb.cbegin(),cdb.cend(), compare_CE()  ));
            invalidate_histo();
            recompute_histo();
            saturated = true;
        } 

        time_total.push_back(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_total).count());
    
        // --------------- performance overview ---- //
        if( n >= 80 ) {
            if( it == 0 )
                std::cerr << std::endl;
            // header
            if( it % 15 == 0 ) {
                std::cerr << "n   It  #B     Bsize  Bvar   A/S      lenbound s.i.len  #Ql      #Q2      #Q3      #nondup  #replaced| L       B       P       Qi      T       Psflops" << std::endl;
            }
            std::cerr << std::left << std::fixed << std::setprecision(0)
                      << std::setw(4) << n
                      << std::setw(4) << it
                      << std::setw(7) << hist_nr_buckets.back() 
                      << std::setw(7) << hist_bsize.back()
                      << std::setw(7) << std::setprecision(0) << hist_bsize_var.back()
                      << std::setw(9) << std::setprecision(2) << double(hist_A.back())/db.size()
                      << std::setw(9) << std::setprecision(3) << hist_lenbound.back()
                      << std::setw(9) << std::setprecision(3) << cdb[saturation_index].len
                      << std::setw(9) << hist_queue_pairs_lift.back()
                      << std::setw(9) << hist_queue_pairs_size.back()
                      << std::setw(9) << hist_queue_triples_size.back()
                      << std::setw(9) << hist_nonduplicates.back()
                      << std::setw(9) << hist_inserted.back()
                      << "| "
                      << std::setprecision(2)
                      << std::setw(8) << time_lenbound.back()/1e6
                      << std::setw(8) << time_bucketing.back()/1e6 
                      << std::setw(8) << time_processing.back()/1e6
                      << std::setw(8) << time_queue.back()/1e6
                      << std::setw(8) << time_total.back()/1e6
                      << std::setw(8) << flops_sieve_processing.back()
                      << std::endl;
            // reset formatting
            std::cerr.copyfmt(std::ios(NULL));
        }

        if( it > 100000 )  {
            std::cerr << "Too many its" << std::endl;
            break;
        }
        if (hist_inserted.back() == 0)
            break;

        assert( cdb.size() == uid_hash_table.hash_table_size() );
        it++;
    }
}
