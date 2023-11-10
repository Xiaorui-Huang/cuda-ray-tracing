#ifndef READ_JSON_H
#define READ_JSON_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Implementation
#include "Float3d.cuh"

#include "Camera.h"
#include "Material.h"

#include "Light.cuh"
#include "Object.cuh"
#include "Plane.cuh"
#include "Sphere.cuh"
#include "Triangle.cuh"

#include "dirname.h"
#include "readSTL.h"
#include <cassert>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

// Forward declaration
struct Object;
struct Light;

// for convenience
using json = nlohmann::json;

// Read a scene description from a .json file
//
// Input:
//   filename  path to .json file
// Output:
//   camera  camera looking at the scene
//   objects  list of shared pointers to objects
//   lights  list of shared pointers to lights
inline bool readJson(const std::string &filename,
                     Camera &camera,
                     std::vector<Object> &objects,
                     std::vector<Light> &lights,
                     std::vector<Material> &materials) {
    // Heavily borrowing from
    // https://github.com/yig/graphics101-raycasting/blob/master/parser.cpp

    std::ifstream infile(filename);
    if (!infile)
        return false;
    json j;
    infile >> j;

    auto parseFloat3d = [](const json &j) -> float3d { return float3d(j[0], j[1], j[2]); };
    // parse a vector
    auto parseCamera = [&parseFloat3d](const json &j, Camera &camera) {
        assert(j["type"] == "perspective" && "Only handling perspective cameras");
        camera.d = j["focal_length"].get<float>();
        camera.e = parseFloat3d(j["eye"]);
        camera.v = parseFloat3d(j["up"]).normalized();
        camera.w = -parseFloat3d(j["look"]).normalized();
        camera.u = camera.v.cross(camera.w);
        camera.height = j["height"].get<float>();
        camera.width = j["width"].get<float>();
    };
    parseCamera(j["camera"], camera);

    std::unordered_map<std::string, std::shared_ptr<Material>> materialsMap;
    auto parseMaterialsMap =
        [&parseFloat3d](const json &j,
                      std::unordered_map<std::string, std::shared_ptr<Material>> &materialsMap) {
            materialsMap.clear();
            for (const json &jmat : j) {
                std::string name = jmat["name"];
                std::shared_ptr<Material> material(new Material());
                material->ka = parseFloat3d(jmat["ka"]);
                material->kd = parseFloat3d(jmat["kd"]);
                material->ks = parseFloat3d(jmat["ks"]);
                material->km = parseFloat3d(jmat["km"]);
                material->phong_exponent = jmat["phong_exponent"];
                materialsMap[name] = material;
            }
        };
    parseMaterialsMap(j["materials"], materialsMap);

    auto parseLights = [&parseFloat3d](const json &j, std::vector<Light> &lights) {
        lights.clear();
        for (const json &jlight : j) {
            if (jlight["type"] == "directional") {
                DirectionalLight dirLight;
                dirLight.direction = parseFloat3d(jlight["direction"]).normalized();
                lights.push_back(Light(dirLight, parseFloat3d(jlight["color"])));
            } else if (jlight["type"] == "point") {
                PointLight pointLight;
                pointLight.position = parseFloat3d(jlight["position"]);
                lights.push_back(Light(pointLight, parseFloat3d(jlight["color"])));
            }
        }
    };
    parseLights(j["lights"], lights);

    auto parseObjects = [&parseFloat3d, &filename, &materialsMap,
                         &materials](const json &j, std::vector<Object> &objects) {
        objects.clear();
        for (const json &jobj : j) {
            if (jobj.count("material")) {
                if (materialsMap.count(jobj["material"])) {
                    materials.push_back(*materialsMap[jobj["material"]]);
                }
            }
            if (jobj["type"] == "sphere") {
                Sphere sphere;
                sphere.center = parseFloat3d(jobj["center"]);
                sphere.radius = jobj["radius"].get<float>();
                objects.push_back(Object(sphere));
            } else if (jobj["type"] == "plane") {
                Plane plane;
                plane.point = parseFloat3d(jobj["point"]);
                plane.normal = parseFloat3d(jobj["normal"]).normalized();
                objects.push_back(Object(plane));
            } else if (jobj["type"] == "triangle") {
                Triangle tri =
                    Triangle(parseFloat3d(jobj["corners"][0]), parseFloat3d(jobj["corners"][1]),
                             parseFloat3d(jobj["corners"][2]));
                objects.push_back(Object(tri));
            } else if (jobj["type"] == "soup") {
                std::vector<std::vector<float>> V;
                std::vector<std::vector<float>> F;
                std::vector<std::vector<int>> N;
                {
#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR std::string("\\")
#else
#define PATH_SEPARATOR std::string("/")
#endif
                    const std::string stl_path = jobj["stl"];
                    igl::readSTL(igl::dirname(filename) + PATH_SEPARATOR + stl_path, V, F, N);
                }
                // std::shared_ptr<TriangleSoup> soup(new TriangleSoup());
                for (int f = 0; f < F.size(); f++) {
                    Triangle tri = Triangle(float3d(V[F[f][0]][0], V[F[f][0]][1], V[F[f][0]][2]),
                                            float3d(V[F[f][1]][0], V[F[f][1]][1], V[F[f][1]][2]),
                                            float3d(V[F[f][2]][0], V[F[f][2]][1], V[F[f][2]][2]));
                    // assigne material index
                    objects.push_back(Object(tri));
                    objects.back().materialIndex = materials.size() - 1;
                }
            }

            objects.back().materialIndex = materials.size() - 1;
        }
    };
    parseObjects(j["objects"], objects);
    return true;
};

#endif
