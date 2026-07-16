# Compatibility between versioned HIP runtime entry points.

const libhip_handle = Ref{Ptr{Cvoid}}(C_NULL)

function has_hip_symbol(name::Symbol)
    handle = libhip_handle[]
    if handle == C_NULL
        handle = AMDGPU.Libdl.dlopen(libhip)
        libhip_handle[] = handle
    end
    AMDGPU.Libdl.dlsym_e(handle, name) != C_NULL
end

# DTK exports ROCm 6-compatible device properties through the `_v2` entry
# point. Its structure extends the R0600 layout by one deprecated field and
# trailing alignment, while all fields consumed by AMDGPU retain their offsets.
const HIP_DEVICE_PROP_R0600_COMPAT_SIZE = 1472
const HIP_DEVICE_PROP_V2_COMPAT_SIZE = 1480

struct hipDeviceProp_tV2Compat
    data::NTuple{HIP_DEVICE_PROP_V2_COMPAT_SIZE,UInt8}
end

function check_device_properties_v2_compat_layout()
    r0600_size = sizeof(hipDeviceProp_tR0600)
    r0600_size == HIP_DEVICE_PROP_R0600_COMPAT_SIZE ||
        error("Unsupported hipDeviceProp_tR0600 layout: expected " *
              "$(HIP_DEVICE_PROP_R0600_COMPAT_SIZE) bytes, got $r0600_size")

    v2_size = sizeof(hipDeviceProp_tV2Compat)
    v2_size == HIP_DEVICE_PROP_V2_COMPAT_SIZE ||
        error("Invalid hipDeviceProp_tV2Compat layout: expected " *
              "$(HIP_DEVICE_PROP_V2_COMPAT_SIZE) bytes, got $v2_size")
    return
end

function device_properties!(prop, device_id)
    if has_hip_symbol(:hipGetDevicePropertiesR0600)
        return hipGetDevicePropertiesR0600(prop, device_id)
    end
    has_hip_symbol(:hipGetDeviceProperties_v2) ||
        error("HIP runtime exports neither hipGetDevicePropertiesR0600 nor " *
              "hipGetDeviceProperties_v2")

    check_device_properties_v2_compat_layout()
    compat = Ref{hipDeviceProp_tV2Compat}()
    status = @check @gcsafe_ccall(
        libhip.hipGetDeviceProperties_v2(
            compat::Ptr{hipDeviceProp_tV2Compat}, device_id::Cint)::hipError_t)
    GC.@preserve prop compat begin
        dst = Base.unsafe_convert(Ptr{hipDeviceProp_tR0600}, prop)
        src = Base.unsafe_convert(Ptr{hipDeviceProp_tV2Compat}, compat)
        unsafe_copyto!(Ptr{UInt8}(dst), Ptr{UInt8}(src), sizeof(hipDeviceProp_tR0600))
    end
    return status
end

function choose_device!(device, prop)
    if has_hip_symbol(:hipChooseDeviceR0600)
        return hipChooseDeviceR0600(device, prop)
    end
    has_hip_symbol(:hipChooseDevice_v2) ||
        error("HIP runtime exports neither hipChooseDeviceR0600 nor hipChooseDevice_v2")

    check_device_properties_v2_compat_layout()
    compat = zeros(UInt8, sizeof(hipDeviceProp_tV2Compat))
    GC.@preserve prop compat begin
        src = Base.unsafe_convert(Ptr{hipDeviceProp_tR0600}, prop)
        unsafe_copyto!(pointer(compat), Ptr{UInt8}(src), sizeof(hipDeviceProp_tR0600))
    end
    GC.@preserve compat begin
        @check @gcsafe_ccall(
            libhip.hipChooseDevice_v2(
                device::Ptr{Cint}, compat::Ptr{hipDeviceProp_tV2Compat})::hipError_t)
    end
end
