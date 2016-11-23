//
// Created by yixuan he on 20/11/2016.
//

#ifndef DL_SIMPLETHREADPOOL_H
#define DL_SIMPLETHREADPOOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
    ThreadPool(size_t);
    ThreadPool();
    void enqueue(std::function<void()> &&f);
    ~ThreadPool();

private:
    void startPool(int size);

    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

#endif //DL_SIMPLETHREADPOOL_H
