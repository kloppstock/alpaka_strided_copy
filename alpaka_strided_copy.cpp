#include <alpaka/alpaka.hpp>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>

// general type definitions
constexpr uint64_t DIMX = 1024;
constexpr uint64_t DIMY = 512;
constexpr uint64_t MAPSIZE = DIMX * DIMY + 8; // image size + header size
const char* PATH = "../../jungfrau/data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat";
struct Frame {
	uint16_t data[MAPSIZE];
};

// general alpaka type definitions
using Dim = alpaka::dim::DimInt<2u>;
using Idx = uint64_t;
using Extent = alpaka::vec::Vec<Dim, uint64_t>;
using Workdiv = alpaka::workdiv::WorkDivMembers<Dim, Extent>;

// accelerator type definitions
using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
using DevHost = alpaka::dev::Dev<Host>;
using DevAcc = alpaka::dev::Dev<Acc>;
using QueueAcc = alpaka::queue::QueueCudaRtNonBlocking;
using PltfHost = alpaka::pltf::Pltf<DevHost>;
using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
using DevBuf = alpaka::mem::buf::Buf<DevAcc, uint16_t, Dim, Idx>;
using HostBuf = alpaka::mem::buf::Buf<DevHost, uint16_t, Dim, Idx>;
using HostView = alpaka::mem::view::ViewSubView<DevHost, uint16_t, Dim, Idx>;

// application specific definitions
struct FramePackage {
	HostBuf data;

	FramePackage(Extent extent, DevHost &host) : data(alpaka::mem::buf::alloc<uint16_t, Idx>(host, extent)) {
		// pin memory to enable fast DMA copy operations
		alpaka::mem::buf::prepareForAsyncCopy(data);
	}
};

int main(int argc, char* argv[]) {
	// alpaka device selection
	DevAcc devAcc = alpaka::pltf::getDevByIdx<PltfAcc>(0u);
	DevHost devHost = alpaka::pltf::getDevByIdx<PltfHost>(0u);
	QueueAcc queue(devAcc);
	
	// determine file size
	struct stat fileStat;
	stat(PATH, &fileStat);
	uint64_t fileSize = fileStat.st_size;
	uint64_t frameCount = fileSize / sizeof(Frame);
	Extent extent(frameCount, MAPSIZE);

	// check for empty file
	if(fileSize == 0) {
		std::cerr << "Error: Empty file or file not found!\n";
		exit(EXIT_FAILURE);
	}

	// allocate data
	FramePackage maps(extent, devHost);

	// open file
	std::ifstream file(PATH, std::ios::in | std::ios::binary);
	if(!file.is_open()) {
		std::cerr << "Error: Couldn't open file!\n";
		exit(EXIT_FAILURE);
	}

	// read file
	file.read(reinterpret_cast<char*>(alpaka::mem::view::getPtrNative(maps.data)), fileSize);
	file.close();

	// read in number of divisions
	int div_input = 2;
	if(argc > 1) {
		div_input = std::atoi(argv[1]);
		if(div_input < 1) {
			std::cerr << "Error: Illegal number of divisions! Defaulting to 2 ...\n";
			div_input = 2;
		}
	}

	// stride settings
	uint64_t divisions = div_input;
	Extent offset(static_cast<uint64_t>(0), static_cast<uint64_t>(0));
	uint64_t splitSize = (MAPSIZE + divisions - 1) / divisions;
	Extent subExtent(frameCount, splitSize);

	// allocate device memory
	DevBuf gpuMem = alpaka::mem::buf::alloc<uint16_t, Idx>(devAcc, subExtent);

	// create source view
	HostView hostView = HostView(maps.data, subExtent, offset);

	// copy data
	alpaka::mem::view::copy(queue, gpuMem, hostView, subExtent);
	
	// syncrhonize
	alpaka::wait::wait(queue);

	return 0;
}
