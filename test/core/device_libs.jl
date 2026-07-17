using Test
using AMDGPU
using GPUCompiler
using LLVM

const Compiler = AMDGPU.Compiler

function devlib_from_ir(name::String, ir::String)
    mod = parse(LLVM.Module, ir)
    try
        mktemp() do path, io
            write(io, mod)
            flush(io)
            Compiler.DevLib(name, path)
        end
    finally
        dispose(mod)
    end
end

@testset "Device-library selection" begin
    GPUCompiler.JuliaContext() do ctx
        mktempdir() do artifact_dir
            mktempdir() do isa_dir
                artifact_file = joinpath(artifact_dir, "oclc_isa_version_900.bc")
                isa_file = joinpath(isa_dir, "oclc_isa_version_936.bc")
                touch(artifact_file)
                touch(isa_file)

                @test Compiler._locate_lib(
                    "oclc_isa_version_936", (isa_dir, artifact_dir)) == isa_file
                @test Compiler._locate_lib(
                    "oclc_isa_version_900", (isa_dir, artifact_dir)) == artifact_file
                @test isnothing(Compiler._locate_lib("oclc_isa_version_938", (isa_dir,)))
            end
        end

        provider_ir = """
            @provided_global = global i32 7

            define i32 @provided_function() {
              ret i32 1
            }

            @provided_alias = alias i32 (), ptr @provided_function

            define ptr @ifunc_resolver() {
              ret ptr @provided_function
            }

            @provided_ifunc = ifunc i32 (), ptr @ifunc_resolver
            """

        lib = devlib_from_ir("alias-provider", provider_ir)
        target = parse(LLVM.Module, """
            declare i32 @provided_alias()
            """)

        @test Compiler.load_and_link!(lib, target, Set(["provided_alias"]))
        @test occursin("@provided_alias = alias", string(target))
        @test issubset(
            Set(["provided_function", "provided_global", "provided_alias",
                 "provided_ifunc"]),
            lib.provided_symbols)
        dispose(target)

        lib = devlib_from_ir("global-provider", provider_ir)
        target = parse(LLVM.Module, "@provided_global = external global i32")
        Compiler._link_referenced_device_libs!(target, (lib,))
        @test "provided_global" ∉ Compiler._undefined_symbol_names(target)
        dispose(target)

        lib = devlib_from_ir("unused-provider", provider_ir)
        target = parse(LLVM.Module, "declare i32 @missing_function()")
        @test !Compiler.load_and_link!(lib, target, Set(["missing_function"]))
        @test !occursin("define i32 @provided_function", string(target))
        dispose(target)

        dependency = devlib_from_ir("dependency", """
            define i32 @dependency() {
              ret i32 2
            }
            """)
        entry = devlib_from_ir("entry", """
            declare i32 @dependency()

            define i32 @entry() {
              %value = call i32 @dependency()
              ret i32 %value
            }
            """)
        target = parse(LLVM.Module, """
            declare i32 @entry()

            define i32 @kernel() {
              %value = call i32 @entry()
              ret i32 %value
            }
            """)

        # Put the dependency first so linking entry requires another scan.
        Compiler._link_referenced_device_libs!(target, (dependency, entry))
        unresolved = Compiler._undefined_symbol_names(target)
        @test "entry" ∉ unresolved
        @test "dependency" ∉ unresolved
        dispose(target)

        target = parse(LLVM.Module, """
            @provided_global = external global i32
            @unused_global = external global i32

            declare i32 @provided_function()
            declare i32 @unused_function()

            define i32 @kernel() {
              %global = load i32, ptr @provided_global
              %value = call i32 @provided_function()
              %result = add i32 %global, %value
              ret i32 %result
            }
            """)
        referenced = Compiler._referenced_undefined_symbol_names(target)
        @test "provided_global" in referenced
        @test "unused_global" ∉ referenced
        @test "provided_function" in referenced
        @test "unused_function" ∉ referenced
        dispose(target)
    end
end
