import AMDGPU: libdevice_libs, libdevice_isa_libs

function _locate_lib(file, directories)
    for directory in directories
        isempty(directory) && continue
        for suffix in (".bc", ".amdgcn.bc")
            file_path = joinpath(directory, file * suffix)
            ispath(file_path) && return file_path
        end
    end
    return nothing
end

locate_lib(file) = _locate_lib(file, (libdevice_libs,))
locate_isa_lib(file) = _locate_lib(file, (libdevice_isa_libs, libdevice_libs))

mutable struct DevLib
    name::String
    path::String
    data::Vector{UInt8}
    # Cached once so kernels that do not use this library avoid reparsing its bitcode.
    provided_symbols::Union{Nothing, Set{String}}

    DevLib(name::String, path::String) = new(name, path, read(path), nothing)
    DevLib(name::String, ::Nothing) = new(name, "", UInt8[], nothing)
end

const DEVICE_LIBS::Dict{String, DevLib} = Dict{String, DevLib}()

function _add_global_alias_names!(names::Set{String}, mod::LLVM.Module)
    # LLVM.jl does not expose a high-level iterator for module aliases.
    alias_ref = LLVM.API.LLVMGetFirstGlobalAlias(mod)
    while alias_ref != C_NULL
        len = Ref{Csize_t}()
        name_ptr = LLVM.API.LLVMGetValueName2(alias_ref, len)
        len[] > 0 && push!(names, unsafe_string(Ptr{UInt8}(name_ptr), len[]))
        alias_ref = LLVM.API.LLVMGetNextGlobalAlias(alias_ref)
    end
    return names
end

function _add_global_ifunc_names!(names::Set{String}, mod::LLVM.Module)
    ifunc_ref = LLVM.API.LLVMGetFirstGlobalIFunc(mod)
    while ifunc_ref != C_NULL
        len = Ref{Csize_t}()
        name_ptr = LLVM.API.LLVMGetValueName2(ifunc_ref, len)
        len[] > 0 && push!(names, unsafe_string(Ptr{UInt8}(name_ptr), len[]))
        ifunc_ref = LLVM.API.LLVMGetNextGlobalIFunc(ifunc_ref)
    end
    return names
end

function _provided_symbol_names(mod::LLVM.Module)
    names = Set{String}(
        LLVM.name(f) for f in functions(mod) if !isdeclaration(f))
    union!(names,
        (LLVM.name(g) for g in globals(mod)
         if !isdeclaration(g) && !isempty(LLVM.name(g))))
    _add_global_alias_names!(names, mod)
    _add_global_ifunc_names!(names, mod)
    return names
end

function _undefined_symbol_names(mod::LLVM.Module)
    names = Set{String}(
        LLVM.name(f) for f in functions(mod)
        if isdeclaration(f) && !LLVM.isintrinsic(f))
    union!(names,
        (LLVM.name(g) for g in globals(mod)
         if isdeclaration(g) && !isempty(LLVM.name(g))))
    return names
end

function _referenced_undefined_symbol_names(mod::LLVM.Module)
    names = Set{String}(
        LLVM.name(f) for f in functions(mod)
        if isdeclaration(f) && !LLVM.isintrinsic(f) && !isempty(uses(f)))
    union!(names,
        (LLVM.name(g) for g in globals(mod)
         if isdeclaration(g) && !isempty(LLVM.name(g)) && !isempty(uses(g))))
    return names
end

function _link_referenced_device_libs!(mod::LLVM.Module, devlibs)
    unresolved_symbols = _undefined_symbol_names(mod)
    linked_libs = falses(length(devlibs))

    # A linked library can introduce references to another library. Re-scan the
    # destination after every link until no remaining library can resolve a symbol.
    while !isempty(unresolved_symbols)
        linked_a_lib = false
        for (i, devlib) in pairs(devlibs)
            linked_libs[i] && continue
            load_and_link!(devlib, mod, unresolved_symbols) || continue

            linked_libs[i] = true
            unresolved_symbols = _undefined_symbol_names(mod)
            linked_a_lib = true
            break
        end
        linked_a_lib || break
    end
    return
end

function link_device_libs!(
    target::GCNCompilerTarget, mod::LLVM.Module;
    wavefrontsize64::Bool,
)
    isnothing(libdevice_libs) && return

    # 1. Preserve module-level metadata carried by hip.bc. It currently has no
    # provider symbols, so a symbol-driven selector cannot discover it.
    devlib = get!(DEVICE_LIBS, "hip") do
        DevLib("hip", locate_lib("hip"))
    end
    load_and_link!(devlib, mod)

    # 2. Load symbol-providing libraries.
    lib_names = ("hc", "irif", "ockl", "opencl", "ocml")
    if !isempty(_undefined_symbol_names(mod))
        devlibs = map(lib_names) do lib_name
            get!(DEVICE_LIBS, lib_name) do
                DevLib(lib_name, locate_lib(lib_name))
            end
        end
        _link_referenced_device_libs!(mod, devlibs)
    end

    # 3. Load OCLC library.
    isa_short = replace(target.dev_isa, "gfx"=>"")
    name = "oclc_isa_version_$isa_short"
    devlib = get!(DEVICE_LIBS, name) do
        DevLib(name, locate_isa_lib(name))
    end
    load_and_link!(devlib, mod)

    # 4. Load OCLC ABI library.
    devlib = get!(DEVICE_LIBS, "oclc_abi") do
        DevLib("oclc_abi", locate_lib("oclc_abi_version_500"))
    end
    load_and_link!(devlib, mod)

    # 5. Load options libraries.
    options = (
        (:finite_only, false),
        (:unsafe_math, false),
        (:correctly_rounded_sqrt, true),
        (:daz_opt, false),
        (:wavefrontsize64, wavefrontsize64))

    for (option, value) in options
        toggle = value ? "on" : "off"
        name = "oclc_$(option)_$(toggle)"
        devlib = get!(DEVICE_LIBS, name) do
            DevLib(name, locate_lib(name))
        end
        load_and_link!(devlib, mod)
    end
end

function close_device_libs!(
    target::GCNCompilerTarget, mod::LLVM.Module;
    wavefrontsize64::Bool,
)
    link_device_libs!(target, mod; wavefrontsize64)

    unresolved = _referenced_undefined_symbol_names(mod)
    isempty(unresolved) && return

    known_symbols = Set{String}()
    for devlib in values(DEVICE_LIBS)
        isnothing(devlib.provided_symbols) && continue
        union!(known_symbols, devlib.provided_symbols)
    end
    residual = intersect(unresolved, known_symbols)
    union!(residual, filter(name -> startswith(name, "__oclc_"), unresolved))
    isempty(residual) || error(
        "Unresolved ROCm device-library symbols after final linking: " *
        join(sort!(collect(residual)), ", "))
    return
end

function load_and_link!(
    devlib::DevLib, mod::LLVM.Module, unresolved_symbols::Union{Nothing, Set{String}}=nothing,
)
    isempty(devlib.path) && return false

    provided_symbols = devlib.provided_symbols
    if !isnothing(unresolved_symbols) && !isnothing(provided_symbols)
        isdisjoint(provided_symbols, unresolved_symbols) && return false
    end

    lib = parse(LLVM.Module, devlib.data)
    if !isnothing(unresolved_symbols) && isnothing(provided_symbols)
        provided_symbols = _provided_symbol_names(lib)
        devlib.provided_symbols = provided_symbols
        if isdisjoint(provided_symbols, unresolved_symbols)
            dispose(lib)
            return false
        end
    end

    inline_attr = EnumAttribute("alwaysinline")
    noinline_attr = EnumAttribute("noinline")

    for f in LLVM.functions(lib)
        fn_name = LLVM.name(f)

        # FIXME: We should be able to inline this, that we can't means
        #        we are inserting calls to it late.
        startswith(fn_name, "__ockl_hsa_signal") && continue

        attrs = function_attributes(f)
        inline = true
        for attr in collect(attrs)
            if kind(attr) == kind(noinline_attr)
                inline = false
                break
            end
        end
        inline && push!(attrs, inline_attr)
    end

    # override triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))
    LLVM.link!(mod, lib)
    return true
end
