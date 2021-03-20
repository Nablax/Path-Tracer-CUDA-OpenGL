//
// Created by LZR on 2021/3/14.
//

#ifndef CUDARAYTRACER_MESH_LOADER_H
#define CUDARAYTRACER_MESH_LOADER_H
#include "OBJ_Loader.hpp"

class MeshLoader{
    MeshLoader(std::string fileName){
        objl::Loader loader;
        loader.LoadFile(fileName);
        auto mesh = loader.LoadedMeshes[0];
        printf("%d", mesh.Vertices.size());
    }
};

#endif //CUDARAYTRACER_MESH_LOADER_H
