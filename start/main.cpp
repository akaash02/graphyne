#include "hnsw.hpp"
#include "io.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr int M = 16;
    constexpr int efConstruction = 200;
    constexpr int ef = 50;
    constexpr int k = 10;

    using clock_type = std::chrono::steady_clock;

    double seconds_since(clock_type::time_point t0)
    {
        return std::chrono::duration<double>(clock_type::now() - t0).count();
    }
}

int main(int argc, char **argv)
{
    const std::string dataDir = (argc > 1) ? argv[1] : "bin";
    const std::string trainPath = dataDir + "/train.bin";
    const std::string testPath = dataDir + "/test.bin";
    const std::string gtPath = dataDir + "/neighbors.bin";

    try
    {
        const auto [n_nodes, dim] = read_header(trainPath);
        std::printf("train: %d x %d  (%.1f MB)\n",
                    n_nodes, dim, double(n_nodes) * dim * sizeof(float) / 1e6);
        std::printf("params: M=%d efConstruction=%d ef=%d k=%d\n\n", M, efConstruction, ef, k);

        std::printf("allocating index...\n");
        HNSW hnsw(M, efConstruction, n_nodes, dim);

        std::printf("loading vectors...\n");
        load_vectors_into(trainPath, hnsw.node_data(), n_nodes, dim);

        std::printf("building index...\n");
        const int progressStep = std::max(1, n_nodes / 20);
        const auto buildStart = clock_type::now();

        for (int i = 0; i < n_nodes; ++i)
        {
            hnsw.addNode(i);

            const int done = i + 1;
            if (done % progressStep == 0 || done == n_nodes)
            {
                const double elapsed = seconds_since(buildStart);
                const double rate = done / elapsed;
                const double eta = (n_nodes - done) / rate;
                std::printf("  %8d / %d  |  %7.0f vec/s  |  elapsed %7.1fs  |  eta %7.1fs\n",
                            done, n_nodes, rate, elapsed, eta);
                std::fflush(stdout);
            }
        }

        const double buildTime = seconds_since(buildStart);
        std::printf("built in %.1fs  (%.0f vec/s)\n\n", buildTime, n_nodes / buildTime);

        NodeBuffer queries = load_vectors(testPath);
        const int nQueries = int(queries.size() / queries.dim());
        if (int(queries.dim()) != dim)
            throw std::runtime_error("query dimensionality does not match train set");

        int32_t gtRows = 0, gtCols = 0;
        const std::vector<int32_t> gt = load_ids(gtPath, gtRows, gtCols);
        if (gtRows != nQueries)
            throw std::runtime_error("ground truth row count does not match query count");
        if (gtCols < k)
            throw std::runtime_error("ground truth has fewer than k columns");

        std::printf("querying: %d queries\n", nQueries);

        std::vector<int> found(size_t(nQueries) * k, -1);
        std::vector<float> q(dim);

        const auto queryStart = clock_type::now();
        for (int i = 0; i < nQueries; ++i)
        {
            const float *row = queries.row(i);
            std::copy(row, row + dim, q.begin());

            const std::vector<int> res = hnsw.searchANN(q, k, ef);
            std::copy(res.begin(), res.end(), found.begin() + size_t(i) * k);
        }
        const double queryTime = seconds_since(queryStart);

        size_t hits = 0;
        for (int i = 0; i < nQueries; ++i)
        {
            const int32_t *truth = gt.data() + size_t(i) * gtCols;
            for (int j = 0; j < k; ++j)
            {
                const int id = found[size_t(i) * k + j];
                if (id < 0)
                    continue;
                if (std::find(truth, truth + k, id) != truth + k)
                    ++hits;
            }
        }

        const double recall = double(hits) / (double(nQueries) * k);
        const double qps = nQueries / queryTime;

        std::printf("\n---- results ----\n");
        std::printf("build     %.1fs  (%.0f vec/s)\n", buildTime, n_nodes / buildTime);
        std::printf("query     %.2fs  (%.0f QPS)\n", queryTime, qps);
        std::printf("recall@%d  %.4f\n", k, recall);
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
