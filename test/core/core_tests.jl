using Test
using AMDGPU
using AMDGPU: HIP, Runtime, Device, Mem

@testset "core" begin

@testset "Functional" begin
    @test AMDGPU.has_rocm_gpu() isa Bool
    @test AMDGPU.functional() isa Bool
end

@testset "HIPDevice" begin
    @testset "Device props" begin
        devices = AMDGPU.devices()
        for (idx, device) in enumerate(devices)
            @test AMDGPU.device_id(device) == idx

            props = HIP.properties(device)
            selected = Ref{Cint}(-1)
            HIP.choose_device!(selected, Ref(props))
            @test 0 <= selected[] < length(devices)

            if HIP.runtime_version() > v"6"
                device_name = HIP.name(device)
                @test length(device_name) > 0
            end

            @test occursin("gfx", HIP.gcn_arch(device))
            @test HIP.wavefrontsize(device) in (32, 64)
        end
    end
end

@testset "ISA parsing" begin
    dev_isa, features = AMDGPU.Compiler.parse_llvm_features("gfx1030")
    @test dev_isa == "gfx1030"
    @test isempty(features)
    dev_isa, features = AMDGPU.Compiler.parse_llvm_features("gfx90a:sramecc+:xnack-")
    @test dev_isa == "gfx90a"
    @test features == "+sramecc,-xnack"
    dev_isa, features = AMDGPU.Compiler.parse_llvm_features("gfx90a:sramecc+:xnack+")
    @test dev_isa == "gfx90a"
    @test features == "+sramecc,+xnack"
    dev_isa, features = AMDGPU.Compiler.parse_llvm_features("gfx90a:xnack-")
    @test dev_isa == "gfx90a"
    @test features == "-xnack"
    dev_isa, features = AMDGPU.Compiler.parse_llvm_features("gfx90a:xnack+")
    @test dev_isa == "gfx90a"
    @test features == "+xnack"
    dev_isa, features = AMDGPU.Compiler.parse_llvm_features("gfx936:sramecc+:xnack-")
    @test dev_isa == "gfx936"
    @test features == "+sramecc,-xnack"
end

@testset "HIP device properties ABI" begin
    @test sizeof(HIP.hipDeviceProp_tR0600) == HIP.HIP_DEVICE_PROP_R0600_COMPAT_SIZE
    @test sizeof(HIP.hipDeviceProp_tV2Compat) == HIP.HIP_DEVICE_PROP_V2_COMPAT_SIZE
    @test HIP.check_device_properties_v2_compat_layout() === nothing
end

@testset "Comparison" begin
    s = AMDGPU.stream()
    @test s == deepcopy(s)

    c = AMDGPU.context()
    @test c == deepcopy(c)

    d = AMDGPU.device()
    @test d == deepcopy(d)
end

end
