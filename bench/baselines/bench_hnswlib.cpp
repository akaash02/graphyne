// hnswlib baseline matching Graphyne's start/main.cpp: M=16, efC=200, k=10, single thread.
// usage: bench_hnswlib <dataDir> <indexPath> build|query
#include "hnswlib/hnswlib.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

template <class T>
static std::vector<T> load(const std::string &p, int &rows, int &cols) {
    std::ifstream f(p, std::ios::binary);
    f.read((char *)&rows, 4); f.read((char *)&cols, 4);
    std::vector<T> v(size_t(rows) * cols);
    f.read((char *)v.data(), v.size() * sizeof(T));
    return v;
}
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    std::string dir = argv[1], idx = argv[2], mode = argv[3];
    int nq, dq, ng, kg;
    auto Q = load<float>(dir + "/test.bin", nq, dq);
    auto G = load<int>(dir + "/neighbors.bin", ng, kg);
    hnswlib::L2Space space(dq);
    hnswlib::HierarchicalNSW<float> *h;
    const int k = 10;

    if (mode == "build") {
        int n, d;
        auto X = load<float>(dir + "/train.bin", n, d);
        h = new hnswlib::HierarchicalNSW<float>(&space, n, 16, 200, 100);
        double t0 = now();
        for (int i = 0; i < n; ++i) {
            h->addPoint(X.data() + size_t(i) * d, i);
            if ((i + 1) % 100000 == 0) { std::printf("  %d  %.0fs\n", i + 1, now() - t0); std::fflush(stdout); }
        }
        double bt = now() - t0;
        std::printf("BUILD hnswlib n=%d %.1fs %.0f vec/s\n", n, bt, n / bt);
        h->saveIndex(idx);
    } else {
        h = new hnswlib::HierarchicalNSW<float>(&space, idx);
    }

    for (int ef : {10, 20, 30, 50, 80, 100, 150, 200, 400}) {
        h->setEf(ef);
        std::vector<int> found(size_t(nq) * k, -1);
        double t0 = now();
        for (int i = 0; i < nq; ++i) {
            auto r = h->searchKnn(Q.data() + size_t(i) * dq, k);
            int j = k - 1;
            while (!r.empty()) { found[size_t(i) * k + j--] = (int)r.top().second; r.pop(); }
        }
        double qt = now() - t0;
        size_t hits = 0;
        for (int i = 0; i < nq; ++i) {
            const int *t = G.data() + size_t(i) * kg;
            for (int j = 0; j < k; ++j)
                if (std::find(t, t + k, found[size_t(i) * k + j]) != t + k) ++hits;
        }
        std::printf("RESULT hnswlib %s ef=%d recall=%.4f qps=%.0f\n", mode.c_str(), ef,
                    double(hits) / (double(nq) * k), nq / qt);
        std::fflush(stdout);
    }
}
