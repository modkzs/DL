//
// Created by yixuan he on 23/11/2016.
//
#include <iostream>
#include "SimpleThreadPool.h"

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    startPool(threads);
}

void ThreadPool::startPool(int size) {
    for (size_t i = 0; i < size; ++i)
        workers.emplace_back(
                [this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                                 [this] { return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                }
        );
}

ThreadPool::ThreadPool() {
    int cores = std::thread::hardware_concurrency();
    startPool(cores);
}

// add new work item to the pool
void ThreadPool::enqueue(std::function<void()> f) {
//    auto task = std::make_shared<std::packaged_task<void> >(f);

//    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace(f);
    }
    condition.notify_one();
//    return res;
}

// the destructor joins all threads
ThreadPool::~ThreadPool() {
    std::cout << "ThreadPool" << std::endl;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker: workers)
        worker.join();
}