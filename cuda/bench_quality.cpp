#include <iostream>

#define DEBUG_BENCHMARK 1
#include "../cuda/GPUStreamGeneral.h"

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



int main(int argc, char **argv) {
    
    size_t n = 128;
    size_t bucket_size = 16*1024;
    size_t bdgl_blocks = 1;
    size_t repeats = 50;
    std::string gpu_bucketer = "bgj1"; 
    size_t multi_bucket = 4;

    int mode = 0; // 0 = collisions, 1 = variation

    if( argc > 1 )
        gpu_bucketer = argv[1];

    if( argc > 2 )
        bucket_size = atoi(argv[2]);

    if( argc > 3 )
        multi_bucket = atoi(argv[3]);

    if( argc > 4 )
        repeats = atoi(argv[4]);

    if( argc > 5 )
        n = atoi(argv[5]);

    if( argc > 6 )
        bdgl_blocks = atoi(argv[6]); 

    if( argc > 7 )
        mode = atoi(argv[7]);

    //std::cerr << argc << ", arguments: " << gpu_bucketer << " " << bucket_size << " " << multi_bucket << " " << bdgl_blocks << std::endl;

    const int device = 0;
    const bool gpu_triple = false; 
    const size_t max_nr_buckets = GPUVECNUM;
    const bool global = true;
    std::vector<Entry> db;
    std::vector<CompressedEntry> cdb;
    const size_t dual_hash_vecs=0;
    const size_t bench_buckets = 2*1024 * (64*1024)/bucket_size;
    const float scale = 1024;
    const float goodip = -0.5;
    const float orthip = std::sqrt(1-goodip*goodip);

    uint32_t b_seed = 3782170233;

    size_t nr_buckets = 3.2 * std::pow(4/3., n/2.) * multi_bucket / bucket_size;
    size_t local_buckets = nr_buckets;

    double avg_bsize = (3.2 * std::pow(4/3., n/2.) * multi_bucket / nr_buckets); 
    size_t bgj1_vecs = 0;

    if( gpu_bucketer == "bdgl" ) {
        local_buckets = std::pow( nr_buckets / float(1<<(bdgl_blocks-1)), 1/float(bdgl_blocks));
        nr_buckets = (1<<(bdgl_blocks-1));
        for( int i = 0; i < bdgl_blocks; i++ ) nr_buckets *= local_buckets;
    } else if( gpu_bucketer == "bgj1") { // bgj1
        nr_buckets -= nr_buckets % 32; 
        bgj1_vecs = repeats*nr_buckets;
    }
 
    const size_t db_size = (mode==0) ? (2*1024*1024) : (1024/multi_bucket * nr_buckets);

    // mu_R
    std::vector<std::vector<float>> mu_R(n, std::vector<float>(n, 0.));
    for( size_t i = 0; i < n; i++ ) {
        mu_R[i][i] = 1./(std::sqrt(n)*scale);
    }
    std::vector<UidType> uid_coeffs(n,0);

    // create database
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.,scale);

    db.resize(db_size + bgj1_vecs);
    cdb.resize(db_size + bgj1_vecs);
    double avglen = 0.;
    for( size_t s = 0; s < db_size + bgj1_vecs; s++ ) {
        // sample x by discretized gaussian
        for( int i = 0; i < n; i++ ) {
            db[s].x[i] = (ZT)(distribution(generator)+.5);
        }
        
        double len = 0.;
        for( int i = 0; i < n; i++ ) {
            len += db[s].x[i] * db[s].x[i] * mu_R[i][i] * mu_R[i][i];
        }
    
        // force angle of 60 degrees between 2i, 2i+1 pair
        if( mode == 0 and s % 2 == 1 and s < db_size ) {
            int ip = 0;
            int lx = 0;
            int ly = 0;
            for( int i = 0; i < n; i++ ) {
                int x = db[s-1].x[i];
                int y = db[s].x[i];
                ip += x*y;
                lx += x*x;
                ly += y*y;
            }
            
            std::vector<float> yperp(n,0.);
            float lyperp = 0.;
            for( int i = 0; i < n; i++ ) {
                int x = db[s-1].x[i];
                int y = db[s].x[i];
                yperp[i] = y - ip/float(lx) * x;
                lyperp += yperp[i] * yperp[i];
            }

            len = 0.;
            for( int i = 0; i < n; i++ ) {
                db[s].x[i] = lround(goodip*db[s-1].x[i] + orthip*yperp[i] * std::sqrt( lx / double(lyperp) ));
                len += db[s].x[i] * db[s].x[i] * mu_R[i][i] * mu_R[i][i];
            }
        }

        avglen += len;

        db[s].len = len;
        db[s].uid = 0;
        cdb[s].i = s;
        cdb[s].len = len;
   
    }
    avglen /= db_size;
    
    //std::cerr << "Bucketing with " << gpu_bucketer << std::endl;

    size_t streams = 1;
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
    
    //std::cerr << "Global init" << std::endl;
    gpu[0]->global_init( mu_R, uid_coeffs );
    
    double cost = 0.;
    double optimal_cost = 0.;
    double var = 0.;
    double expected = 0.;
    std::set<indextype> capt;

    std::uniform_int_distribution<int64_t> unif; 

    size_t signsup = 0;
    size_t signsdown = 0;
        
    for( size_t rep = 0; rep < repeats; rep++ ) {
        b_seed = unif(generator);

        // BUCKETING PREPARE
        //std::cerr << "Bucketing prepare" << std::endl;
        std::vector<double> q;
        if( gpu_bucketer == "bdgl" ) {
            q.resize(n*n);
            // construct random orthonormal transform
            std::vector<double> orth(n*n);
            random_orthogonal (orth, n, unif(generator));

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
       
        std::vector<size_t> b_indices( nr_buckets );
        for( int b = 0; b < nr_buckets; b++ )
            b_indices[b] = ((bgj1_vecs>0)?(db_size + rep*nr_buckets):0) + b;

        //std::cerr << "B_init " << b_seed << std::endl;
        for( size_t s = 0; s < streams; s++ ) {
            gpu[s]->bind( db, cdb);
            gpu[s]->B_init( b_indices, b_seed, q, bdgl_blocks );
        }

        //std::cerr << "Prepare result space" << std::endl;
        size_t vecs_per_bucket = (8*db_size*multi_bucket) / nr_buckets;
        std::vector<indextype> all_bucket_indices(vecs_per_bucket * nr_buckets, 0);
        std::vector<iptype> all_bucket_ips(vecs_per_bucket * nr_buckets);
        std::vector<triple_bucket> t_buckets(nr_buckets);
        
        for( size_t b = 0; b < nr_buckets; b++ ) {
            t_buckets[b].indices = all_bucket_indices.data() + b * vecs_per_bucket;
            t_buckets[b].ips = all_bucket_ips.data() + b * vecs_per_bucket;;
            t_buckets[b].size = 0;
        }

        //std::cerr << "Start bucketing with " << nr_buckets << " buckets of avg size " <<  (3.2 * std::pow(4/3., n/2.) * multi_bucket / nr_buckets)  << std::endl;

        if( gpu_bucketer == "bdgl" or gpu_bucketer == "bgj1" ) {
            size_t chunk_size = 64*1024; 
            for( size_t cdb_start = 0, s = 0; cdb_start < db_size; cdb_start += chunk_size, s++ ) {
                size_t cdb_end = std::min( cdb_start+chunk_size, db_size );
                gpu[s%streams]->B_send_data( cdb_start, cdb_end );
                gpu[s%streams]->B_launch_kernel();
                gpu[s%streams]->B_receive_data( t_buckets, vecs_per_bucket, false );
            }

            for( size_t i = 0; i < streams; i++ ) {
                gpu[i]->B_receive_data( t_buckets, vecs_per_bucket, true );
            }

        }
        else if( gpu_bucketer == "rand" ) {
            std::uniform_int_distribution<indextype> randbucket(0, nr_buckets-1);
            std::vector<indextype> bb(multi_bucket);
            for( size_t cdb_start = 0; cdb_start < db_size; cdb_start++ ) {
                for( int i = 0; i < multi_bucket; i++ ) {
                    bb[i] = randbucket(generator);
                    assert(bb[i] >= 0 and bb[i] < nr_buckets );
                }
                

                // check if all distinct
                std::sort(bb.begin(), bb.end());
                bool distinct = true;
                for( int i = 0; i < multi_bucket-1; i++ )
                    if( bb[i] == bb[i+1] )
                        distinct = false;
                if( !distinct ) {
                    cdb_start--;
                    continue;
                }

                for( int i = 0; i < multi_bucket; i++ ) {
                    size_t ind = t_buckets[bb[i]].size;
                    t_buckets[bb[i]].size++;
                    t_buckets[bb[i]].indices[ind] = cdb_start;
                    t_buckets[bb[i]].ips[ind] = __float2half(0.01);
                }
            }
        }

        //std::cerr << "Processing results " << std::endl;

        // find captures
        
        for( size_t b = 0; b < nr_buckets; b++ ) {
            cost += double(t_buckets[b].size) * double(t_buckets[b].size);
        }
        optimal_cost += double(multi_bucket * db_size) * double(multi_bucket * db_size) / nr_buckets; 

        if( mode == 0 ) { 
            for( size_t b = 0; b < nr_buckets; b++ ) { 
        

                expected += double(t_buckets[b].size)/double(repeats*nr_buckets);
                var += double(t_buckets[b].size) * double(t_buckets[b].size)/(double(repeats*nr_buckets));
                for( int i = 0; i < int(t_buckets[b].size)-1; i++ ) {            
                    signsup += __half2float(t_buckets[b].ips[i])>0;
                    signsdown += __half2float(t_buckets[b].ips[i])<=0;
        
                    if( t_buckets[b].indices[i] % 2 == 0 and t_buckets[b].indices[i+1] == t_buckets[b].indices[i]+1 
                            //and ((__half2float(t_buckets[b].ips[i])<0) == (__half2float(t_buckets[b].ips[i+1])>0) )
                            )
                        capt.insert( t_buckets[b].indices[i]/2 );
                }
            }
            std::cerr << rep << ": " << capt.size() << " / " << db_size/2 << std::endl;
        }
            
        //if( mode == 1 ) {
        //
        //    std::cerr << n << " " << ((gpu_bucketer=="bdgl")?(gpu_bucketer.append(std::to_string(bdgl_blocks))):gpu_bucketer) << " " << avg_bsize << " " << multi_bucket << " " << repeats << " " << (cost/optimal_cost) << std::endl;
        //    size_t sm=0;
        //    for( size_t b= 0; b < nr_buckets; b++ ) {    
        //        std::cout << t_buckets[b].size << std::endl;
        //        sm += t_buckets[b].size;
        //    }
        //    std::cout << db_size << " " << sm << std::endl;
        //}
    }

    //std::cerr << "Variation: " << (var - expected*expected) << std::endl;
    //std::cout << avg_bsize << " " << bdgl_blocks << " " << nr_buckets << " " << (var-expected*expected) << " " << (double(capt.size())/(db_size/2)) << std::endl; 
    
    std::cerr << "signs: " << signsup << " " << signsdown << std::endl;
    std::cout << n << " " << ((gpu_bucketer=="bdgl")?(gpu_bucketer.append(std::to_string(bdgl_blocks))):gpu_bucketer) << " " << avg_bsize << " " << multi_bucket << " " << repeats << " " << (double(capt.size())/(db_size/2)) << " " << (cost/optimal_cost) << std::endl;

    for( int s = 0; s < streams; s++ ) {
        //gpu[s]->P_receive_data( queue, true );
        //gpu[s]->bench_results();
        gpu[s]->free();
        delete gpu[s];
    }

    return 0;
}
