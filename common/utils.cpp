#include "utils.h"
#include <fstream>
#include <iostream>

bool read_vector_from_file(const std::string& filename, std::vector<float>& vec) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    vec.resize(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(vec.data()), size)) {
        std::cerr << "Error: Could not read data from file " << filename << std::endl;
        return false;
    }
    return true;
}
