#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h> // MPI: 包含 MPI 头文件
#include <numeric>
#include <unordered_set>
using namespace std;
using namespace chrono;


enum DataType { BOOL_DATA = 1, STRING_DATA = 2};
void send_vector_of_strings(const std::vector<std::string>& vec, int dest_rank, MPI_Comm comm) {
    // --- 第一阶段：发送元数据 ---

    // 1. 发送 vector 中的字符串数量
    size_t num_strings = vec.size();
    MPI_Send(&num_strings, 1, MPI_UNSIGNED_LONG, dest_rank, 0, comm);//发送给dest_rank号进程

    // 2. 准备并发送每个字符串的长度
    std::vector<size_t> lengths;
    lengths.reserve(num_strings);
    size_t total_chars = 0;
    for (const auto& s : vec) {
        lengths.push_back(s.length());
        total_chars += s.length();
    }
    MPI_Send(lengths.data(), num_strings, MPI_UNSIGNED_LONG, dest_rank, 1, comm);

    // --- 第二阶段：打包并发送实际数据 ---
    
    // 3. 将所有字符串打包到一个连续的 char 缓冲区
    std::string packed_data;
    packed_data.reserve(total_chars);
    for (const auto& s : vec) {
        packed_data.append(s);
    }

    // 4. 发送打包好的数据
    MPI_Send(packed_data.c_str(), packed_data.length(), MPI_CHAR, dest_rank, 2, comm);
}

std::vector<std::string> receive_vector_of_strings(int source_rank, MPI_Comm comm) {
    std::vector<std::string> vec;
    MPI_Status status;

    // --- 第一阶段：接收元数据 ---

    // 1. 接收字符串数量
    size_t num_strings;
    MPI_Recv(&num_strings, 1, MPI_UNSIGNED_LONG, source_rank, 0, comm, &status);

    // 2. 接收每个字符串的长度
    std::vector<size_t> lengths(num_strings);
    MPI_Recv(lengths.data(), num_strings, MPI_UNSIGNED_LONG, source_rank, 1, comm, &status);

    // --- 第二阶段：接收并解包实际数据 ---

    // 3. 计算总字符数并准备接收缓冲区
    size_t total_chars = std::accumulate(lengths.begin(), lengths.end(), (size_t)0);
    std::vector<char> packed_buffer(total_chars);

    // 4. 接收打包好的数据
    MPI_Recv(packed_buffer.data(), total_chars, MPI_CHAR, source_rank, 2, comm, &status);

    // 5. 解包数据，重建 vector<string>
    size_t current_pos = 0;
    for (size_t len : lengths) {
        // 从缓冲区的正确位置和长度构造 string
        vec.emplace_back(packed_buffer.data() + current_pos, len);
        current_pos += len;
    }

    return vec;
}

int main()
{
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 确保至少有两个进程
    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "This program requires at least 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

	double time_hash = 0;
	double time_guess = 0;
	double time_train = 0;
	PriorityQueue q;
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;
    // ------------------- 主进程 (rank 0) 逻辑 -------------------
    if (rank == 0) {
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        
        q.init();
	    cout << "here" << endl;
        
        int curr_num = 0;
	    auto start = system_clock::now();
	    int history = 0;
        bool continue_looping = true;
        
        do {
            q.PopNext(); // 只有 rank 0 在生成
            
            q.total_guesses = q.guesses.size();
            
            if (q.total_guesses - curr_num >= 100000) {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                int generate_n = 10000000;
                if (history + q.total_guesses > generate_n) {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    continue_looping = false;
                }
            }

            if (curr_num > 1000000) {
                int data_type_id = STRING_DATA;
                MPI_Send(&data_type_id, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD);
                send_vector_of_strings(q.guesses, size - 1, MPI_COMM_WORLD);
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }

            if (q.priority.empty()) {
                continue_looping = false;
            }

            if (!continue_looping) {
                int data_type_id = BOOL_DATA;
                MPI_Send(&data_type_id, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD);
                MPI_Send(&continue_looping, 1, MPI_CXX_BOOL, size - 1, 0, MPI_COMM_WORLD);
            }
        
        } while (continue_looping);
        
        // --- 接收最终时间戳并计算总时间 ---
        MPI_Status status;
        long long received_end1_microseconds = 0;
        MPI_Recv(&received_end1_microseconds, 1, MPI_LONG_LONG, size - 1, 100, MPI_COMM_WORLD, &status);
        
        long long end0_microseconds = time_point_cast<microseconds>(system_clock::now()).time_since_epoch().count();
        long long start_microseconds = time_point_cast<microseconds>(start).time_since_epoch().count();
        long long tend = max(end0_microseconds, received_end1_microseconds);
        double total_times = (tend - start_microseconds) / 1000000.0;
        cout << "total time:" << total_times << "seconds" << endl;
    }
    // ------------------- 哈希/破解进程 (rank == size - 1) 逻辑 -------------------
    else if (rank == size - 1) { 
        unordered_set<std::string> test_set;
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count=0;
        string pw;
        while(test_data>>pw) {   
            test_count+=1;
            test_set.insert(pw);
            if (test_count>=1000000) {
                break;
            }
        }
        
        
        vector<string> q_guesses;
        bool continue_next = true;
        
        while(continue_next) { // <-- 简化循环条件
            int data_type_id;
            MPI_Status status;
            MPI_Recv(&data_type_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            if (data_type_id == STRING_DATA) {
                std::vector<std::string> received_guesses = receive_vector_of_strings(0, MPI_COMM_WORLD);
                q_guesses.insert(q_guesses.end(), received_guesses.begin(), received_guesses.end());
            } else if (data_type_id == BOOL_DATA) {
                MPI_Recv(&continue_next, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &status);
            }
            
            // 每次收到数据后立即处理
            auto start_hash_batch = system_clock::now();
            for (size_t i = 0; i < q_guesses.size(); ++i) { // <-- 简化循环，不使用SIMD批处理格式
                if (test_set.count(q_guesses[i])) { // .count() 比 .find() 更直接
                    cracked += 1;
                }
                // MD5Hash a single password or a batch, depending on your needs
            }
            auto end_hash_batch = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash_batch - start_hash_batch);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            q_guesses.clear();
        }

        cout << "Hash time: " << time_hash << " seconds" << endl;
        cout << "Cracked: " << cracked << endl;
        
        long long end1_microseconds = time_point_cast<microseconds>(system_clock::now()).time_since_epoch().count();
        MPI_Send(&end1_microseconds, 1, MPI_LONG_LONG, 0, 100, MPI_COMM_WORLD);
    }
    cout<<"Cracked:"<<cracked<<endl;
	MPI_Finalize();
	return 0;
}