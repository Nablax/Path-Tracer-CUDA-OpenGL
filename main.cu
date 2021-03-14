#include <png_image.h>
#include <ctime>
#include "cuda_check.h"
#include "camera.h"
#include "render_manager.h"
#include "material.h"
#include "bvh.h"
#include "cuda2gl.h"
#include "macros.h"

surface<void,cudaSurfaceType2D> surf;
camera *devCamera, *hostCamera;
RenderManager *world;
size_t frameBufferSize = globalvar::kFrameHeight * globalvar::kFrameWidth * sizeof(vec3);
curandState *devStates;
BVHNode *lbvhArrayDevice;
dim3 blocks(globalvar::kBlockX, globalvar::kBlockY);
dim3 threads(globalvar::kThreadX, globalvar::kThreadY);
double deltaTime = 0;

__device__ color ray_color(const Ray& r, RenderManager *world, int depth, curandState *randState) {
    hit_record rec;
    Ray curRay = r;
    //printf("in ray color\n");
    color attenuation(1, 1, 1);
    while(depth-- > 0){
        if (world->hitBvh(curRay, 0.001f, globalvar::kInfinityGPU, rec)) {
            color nextAttenuation;
            if (world->mMaterials[rec.matID].scatter(curRay, rec, nextAttenuation, curRay, randState))
                attenuation *= nextAttenuation;
            else attenuation = vec3();
        }
        else{
            vec3 unit_direction = vectorgpu::normalize(curRay.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            return ((1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0)) * attenuation;
        }
    }
    return {};
}

__global__ void copyObjMatsToDevice(RenderManager* world, CudaObj* myObj, int objSize, Material* myMats, int matSize){
    world->mObjects = myObj;
    world->mMaterials = myMats;
    world->mMatMaxSize = world->mMatLastIdx = matSize;
    world->mObjMaxSize = world->mObjLastIdx = objSize;
}
__global__ void copyBVHToDevice(RenderManager* world, BVHNode* bvh){
    world->bvh = bvh;
#ifdef TEST
    world->printBvh();
#endif
}


void generateTestWorldOnHost(){
    std::vector<CudaObj> myObj;
    std::vector<Material> myMats;
    aabb maxBox;

//    myObj.emplace_back(point3(0, -1000, 0), 1000.0f, myMats.size());
//    myMats.emplace_back(color(0.5, 0.5, 0.5));
//    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);

//    myObj.emplace_back(point3(1005, 0, 0), 1000.0f, myMats.size());
//    myMats.emplace_back(color(1, 0, 0));
//    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);


    myObj.emplace_back(point3(2, 1, 0), 1.0f, myMats.size());
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);
    myObj.emplace_back(point3(2, 1, 0), -0.9f, myMats.size());
    myMats.emplace_back(1.5f);


    myObj.emplace_back(point3(-2, 1, 0), 1.0f, myMats.size());
    myMats.emplace_back(color(1, 0, 0.4));
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);

    myObj.emplace_back(point3(0, 1, 0), 1.0f, myMats.size());
    myMats.emplace_back(color(0.7, 0.6, 0.5), 0.0f);
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);

    myObj.emplace_back(point3(2, -1, 0), 1.0f, myMats.size());
    myMats.emplace_back(color(1, 0, 0));
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);

    myObj.emplace_back(point3(1005, 0, 0), 1000.0f, myMats.size());
    myMats.emplace_back(color(0, 0, 1));
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);

    CudaObj *myObjCuda;
    Material *myMatsCuda;
    checkCudaErrors(cudaMalloc((void**)&myObjCuda, myObj.size() * sizeof(CudaObj)));
    checkCudaErrors(cudaMalloc((void**)&myMatsCuda, myMats.size() * sizeof(Material)));

    checkCudaErrors(cudaMemcpy(myObjCuda, myObj.data(), myObj.size() * sizeof(CudaObj), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(myMatsCuda, myMats.data(), myMats.size() * sizeof(Material), cudaMemcpyHostToDevice));

    copyObjMatsToDevice<<<1, 1>>>(world, myObjCuda, myObj.size(), myMatsCuda, myMats.size());
    checkCudaErrors(cudaDeviceSynchronize());
    lbvh::buildBVH(myObj, myObjCuda, maxBox, lbvhArrayDevice);
    copyBVHToDevice<<<1, 1>>>(world, lbvhArrayDevice);
}

void generateRandomWorldOnHost(){
    std::vector<CudaObj> myObj;
    std::vector<Material> myMats;
    aabb maxBox;

    myObj.emplace_back(point3(0, -1000, 0), 1000.0f, myMats.size());
    myMats.emplace_back(color(0.5, 0.5, 0.5));
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);

    int sampleNum = 0;

    for(int i = -sampleNum; i < sampleNum; i++){
        for(int j = -sampleNum; j < sampleNum; j++){
            float choose_mat = randomUniformOnHost();
            point3 center(i + 0.9f * randomUniformOnHost(), 0.2, j + 0.9f * randomUniformOnHost());
            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                auto rand1 = vec3(randomUniformOnHost(), randomUniformOnHost(), randomUniformOnHost());
                auto rand2 = vec3(randomUniformOnHost(), randomUniformOnHost(), randomUniformOnHost());
                if(choose_mat < 0.8){
                    auto albedo = rand1 * rand2;
                    myObj.emplace_back(center, 0.2f, myMats.size());
                    myMats.emplace_back(albedo);
                }
                else if(choose_mat < 0.95){
                    auto albedo = rand1 / 2 + vec3(0.5f, 0.5f, 0.5f);
                    float fuzz = rand2.x() / 2;
                    myObj.emplace_back(center, 0.2f, myMats.size());
                    myMats.emplace_back(albedo, fuzz);
                }
                else{
                    myObj.emplace_back(center, 0.2f, myMats.size());
                    myMats.emplace_back(1.5f);
                }
                maxBox.unionBoxInPlace(myObj.back().mBoundingBox);
            }
        }
    }
    myObj.emplace_back(point3(4, 1, 0), 1.0f, myMats.size());
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);
    myObj.emplace_back(point3(4, 1, 0), -0.9f, myMats.size());
    myMats.emplace_back(1.5f);

    myObj.emplace_back(point3(-4, 1, 0), 1.0f, myMats.size());
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);
    myMats.emplace_back(color(1, 0, 0.4));

    myObj.emplace_back(point3(0, 1, 0), 1.0f, myMats.size());
    maxBox.unionBoxInPlace(myObj.back().mBoundingBox);
    myMats.emplace_back(color(0.7, 0.6, 0.5), 0.0f);

    CudaObj *myObjCuda;
    Material *myMatsCuda;
    checkCudaErrors(cudaMalloc((void**)&myObjCuda, myObj.size() * sizeof(CudaObj)));
    checkCudaErrors(cudaMalloc((void**)&myMatsCuda, myMats.size() * sizeof(Material)));

    checkCudaErrors(cudaMemcpy(myObjCuda, myObj.data(), myObj.size() * sizeof(CudaObj), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(myMatsCuda, myMats.data(), myMats.size() * sizeof(Material), cudaMemcpyHostToDevice));

    copyObjMatsToDevice<<<1, 1>>>(world, myObjCuda, myObj.size(), myMatsCuda, myMats.size());
    checkCudaErrors(cudaDeviceSynchronize());
    lbvh::buildBVH(myObj, myObjCuda, maxBox, lbvhArrayDevice);
    copyBVHToDevice<<<1, 1>>>(world, lbvhArrayDevice);
}

__global__ void clearWorld(RenderManager *world){
    delete world;
}

__global__ void initRandom(curandState *randState, int maxWidth, int maxHeight, int seed){
    unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    unsigned curPixel = row * maxWidth + col;
    curand_init(seed, curPixel, 0, &randState[curPixel]);
}

__global__ void render(vec3 *frameBuffer, int maxWidth, int maxHeight, int spp, int maxDepth,
                       camera *myCamera,
                       RenderManager *world, curandState *randState){
    //printf("%d %d %d %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x, blockDim.y, threadIdx.y, threadIdx.y);
    unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    //printf("%d %d %d %d\n", row, col, maxWidth, maxHeight);
    unsigned curPixel = row * maxWidth + col;
    float maxWidthInv = 1.0f / maxWidth, maxHeightInv = 1.0f / maxHeight, sppInv = 1.0f / spp;
    color finalColor = color();
    for(int i = 0; i < spp; i++){
        float u = (col + curand_uniform(&randState[curPixel])) * maxWidthInv;
        float v = (row + curand_uniform(&randState[curPixel])) * maxHeightInv;
        Ray r = myCamera->get_ray(u, v, randState);
        //printf("%f %f %f\n", u, v, myCamera->fl);
        finalColor += ray_color(r, world, maxDepth, &randState[curPixel]);
    }
    float r = sqrtf(finalColor.r() * sppInv);
    float g = sqrtf(finalColor.g() * sppInv);
    float b = sqrtf(finalColor.b() * sppInv);
    frameBuffer[curPixel] = vec3(r, g, b);
}

union ColorRGBA
{
    unsigned color;
    struct {
        unsigned  r  : 8;
        unsigned  g  : 8;
        unsigned  b  : 8;
        unsigned  a : 8;
    };
};


bool goRender = true;

__global__ void renderBySurface(int maxWidth, int maxHeight, int spp, int maxDepth,
                       camera *myCamera,
                       RenderManager *world, curandState *randState, bool goRender){
    if(!goRender) return;
    //printf("%d %d %d %d %d %d\n", blockDim.x, threadIdx.x, threadIdx.x, blockDim.y, threadIdx.y, threadIdx.y);
    unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= maxHeight || col >= maxWidth) return;
    //printf("%d %d %d %d\n", row, col, maxWidth, maxHeight);
    unsigned curPixel = row * maxWidth + col;
    float maxWidthInv = 1.0f / maxWidth, maxHeightInv = 1.0f / maxHeight, sppInv = 1.0f / spp;

    vec3 color = vec3();

    union ColorRGBA rgba;

    for(int i = 0; i < spp; i++){
        float u = (col + curand_uniform(&randState[curPixel])) * maxWidthInv;
        float v = (row + curand_uniform(&randState[curPixel])) * maxHeightInv;
        Ray r = myCamera->get_ray(u, v, randState);
        //printf("%f %f %f\n", u, v, myCamera->fl);
        color += ray_color(r, world, maxDepth, &randState[curPixel]);
    }
    rgba.r = sqrtf(color.r() * sppInv) * 255;
    rgba.g = sqrtf(color.g() * sppInv) * 255;
    rgba.b = sqrtf(color.b() * sppInv) * 255;
    rgba.a = 255;

    surf2Dwrite(rgba.color,
                surf,
                col * sizeof(rgba),
                row,
                cudaBoundaryModeZero);
}

static void fpsCount(GLFWwindow* window)
{
    static double lastFrame  = 0.0;
    static int frameCount = 0;
    const double currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    if (deltaTime > 0.5)
    {
        lastFrame = currentFrame;
        const double fps = (double)frameCount / deltaTime;
        int  width, height;
        char tmp[64];
        glfwGetFramebufferSize(window,&width,&height);
        sprintf_s(tmp,64,"(%u x %u) - FPS: %.2f", width, height, fps);
        glfwSetWindowTitle(window,tmp);
        frameCount = 0;
    }
    frameCount++;
}

void myGlInit(GLFWwindow** window, const int width, const int height){
    if (!glfwInit())
        exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_DEPTH_BITS,            0);
    glfwWindowHint(GLFW_STENCIL_BITS,          0);

    //glfwWindowHint(GLFW_SRGB_CAPABLE,          GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);

    *window = glfwCreateWindow(width,height,"GLFW / CUDA Interop",NULL,NULL);
    if (*window == NULL)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(*window);
    // set up GLAD
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    // ignore vsync for now
    glfwSwapInterval(0);
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
        glfwSetWindowShouldClose(window, true);
        return;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        hostCamera->processKeyboard(FORWARD, deltaTime);
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        hostCamera->processKeyboard(BACKWARD, deltaTime);
    else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        hostCamera->processKeyboard(LEFT, deltaTime);
    else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        hostCamera->processKeyboard(RIGHT, deltaTime);
    else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        hostCamera->processKeyboard(UP, deltaTime);
    else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        hostCamera->processKeyboard(DOWN, deltaTime);
    else return;
    checkCudaErrors(cudaMemcpy(devCamera, hostCamera, sizeof(camera), cudaMemcpyHostToDevice));
}

void initWorldStates(){
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(RenderManager)));
    hostCamera = new camera(
            vec3 (13,2,3),
            vec3(0,0,0), 20,
            globalvar::kAspectRatio,
            0, 10, 0.0, 1.0);
    checkCudaErrors(cudaMalloc((void **)&devCamera, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(devCamera, hostCamera, sizeof(camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&devStates, frameBufferSize * sizeof(curandState)));
    srand(time(nullptr));
    int seed = rand();
    initRandom<<<blocks, threads>>>(devStates, globalvar::kFrameWidth, globalvar::kFrameHeight, seed);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //generateRandomWorld<<<1, 1>>>(world, devStates);
    //generateRandomWorldOnHost();
#ifdef TEST
    generateTestWorldOnHost();
    hostCamera = new camera(
            vec3 (0,1,15),
            vec3(0,0,0), 20,
            globalvar::kAspectRatio,
            0, 10, 0.0, 1.0);
    checkCudaErrors(cudaMemcpy(devCamera, hostCamera, sizeof(camera), cudaMemcpyHostToDevice));
#else
    generateRandomWorldOnHost();
#endif
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void clearWorldStates(){
    clearWorld<<<1, 1>>>(world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(devStates));
    checkCudaErrors(cudaFree(devCamera));
    checkCudaErrors(cudaFree(lbvhArrayDevice));
    cudaDeviceReset();
}

void renderToPng(){
    PngImage png(globalvar::kFrameWidth, globalvar::kFrameHeight);
    vec3 *frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));
    initWorldStates();

    printf("Start rendering!\n");
    std::clock_t start = std::clock();
    render<<<blocks, threads>>>(frameBuffer, globalvar::kFrameWidth, globalvar::kFrameHeight,
                                globalvar::kSpp, globalvar::kMaxDepth,
                                devCamera, world, devStates);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Time Cost: "<< duration <<'\n';
    for(int row = 0; row < globalvar::kFrameHeight; row++){
        for(int col = 0; col < globalvar::kFrameWidth; col++){
            int curPixel = row * globalvar::kFrameWidth + col;
            //std::cerr << frameBuffer[curPixel] << '\n';
            png.saveColor(frameBuffer[curPixel], globalvar::kFrameHeight - row - 1, col);
        }
    }
    png.write("../output2/debug.png");
    checkCudaErrors(cudaFree(frameBuffer));
    clearWorldStates();
}

void renderToGL(){
    initWorldStates();
    GLFWwindow *window;
    myGlInit(&window, globalvar::kFrameWidth, globalvar::kFrameHeight);

    cudaStream_t stream;
    cudaEvent_t  event;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream,cudaStreamDefault));
    checkCudaErrors(cudaEventCreateWithFlags(&event,cudaEventBlockingSync));

    Cuda2Gl* interop = new Cuda2Gl(2);

    int width, height;
    glfwGetFramebufferSize(window,&width,&height);
    interop->updateFrameSize(width,height);
    glfwSetWindowUserPointer(window,interop);
    while (!glfwWindowShouldClose(window))
    {
        fpsCount(window);
        processInput(window);
        interop->getFrameSize(&width,&height);
        interop->mapGraphicResource(stream);
        cudaBindSurfaceToArray(surf, interop->getCudaArray());

        renderBySurface<<<blocks, threads>>>(globalvar::kFrameWidth, globalvar::kFrameHeight,
                                             globalvar::kSpp, globalvar::kMaxDepth,
                                             devCamera, world, devStates, goRender);

        interop->unMapGraphicResource(stream);
        interop->blitFramebuffer();
        interop->swapBuffer();
        glfwSwapBuffers(window);
        glfwPollEvents(); // glfwWaitEvents();
    }
    delete interop;
    glfwDestroyWindow(window);
    glfwTerminate();
    clearWorldStates();
}

int main()
{
    renderToPng();
}

