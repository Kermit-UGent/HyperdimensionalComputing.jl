@testset "encoding" begin
    hvs = BinaryHV.(
        [
            Bool.([1, 0, 0, 0, 0]),
            Bool.([1, 1, 0, 0, 0]),
            Bool.([1, 1, 1, 0, 0]),
            Bool.([1, 1, 1, 1, 0]),
            Bool.([1, 1, 1, 1, 1]),
        ]
    )

    @testset "multiset" begin
        @test multiset(hvs).v == Bool.([1, 1, 1, 0, 0])
    end

    @testset "multibind" begin
        @test multibind(hvs).v == Bool.([1, 0, 1, 0, 1])
    end

    @testset "bundlesequence" begin
        @test bundlesequence(hvs).v == ones(Bool, 5)
        @test_throws AssertionError bundlesequence([first(hvs)])
    end

    @testset "bindsequence" begin
        @test bindsequence(hvs).v == ones(Bool, 5)
        @test_throws AssertionError bindsequence([first(hvs)])
    end

    @testset "hashtable" begin
        @test hashtable(hvs, hvs) == zeros(Bool, 5)
        @test_throws AssertionError hashtable(hvs, hvs[1:2])
    end

    @testset "crossproduct" begin
        @test crossproduct(hvs, hvs) == zeros(Bool, 5)
    end

    @testset "ngrams" begin
        @test ngrams(hvs).v == Bool.([0, 1, 0, 0, 1])
        @test ngrams(hvs) == bundle([hvs[1] * ρ(hvs[2]) * ρ(hvs[3], 2), hvs[2] * ρ(hvs[3]) * ρ(hvs[4], 2), hvs[3] * ρ(hvs[4]) * ρ(hvs[5], 2)])
        @test_throws AssertionError ngrams(hvs, 0)
        @test_throws AssertionError ngrams(hvs, length(hvs) + 1)
    end

    @testset "graph" begin
        s = [1, 3, 4, 2, 5]
        t = [3, 4, 2, 1, 4]
        @test graph(hvs[s], hvs[t]) == Bool.([0, 0, 0, 0, 0])
        @test graph(hvs[s], hvs[t]; directed = true) == Bool.([1, 0, 0, 1, 0])
        @test_throws AssertionError graph(hvs[s], hvs[[1, 2, 3]])
    end

    @testset "levels" begin
        numvals = 0:0.1:2pi
        levels = level(BinaryHV(100), numvals)

        @test length(levels) == length(numvals)
        @test eltype(levels) <: BinaryHV

        encoder, decoder = convertlevel(levels, numvals)
        hv = encoder(1.467)
        @test hv isa BinaryHV
        x = decoder(hv)
        @test 1 ≤ x ≤ 2

        # regression (TODO §1.4): the instance-based decodelevel/convertlevel
        # path used to throw a MethodError caused by kwarg forwarding
        dec = decodelevel(BinaryHV(; D = 100, seed = 3), numvals)
        @test dec isa Function
        enc2, dec2 = convertlevel(BinaryHV(; D = 100, seed = 3), numvals)
        @test enc2(1.0) isa BinaryHV
        @test minimum(numvals) ≤ dec2(enc2(1.0)) ≤ maximum(numvals)
        # keywords also forward through the shared-ladder path
        enc3, dec3 = convertlevel(levels, numvals; testbound = true)
        @test dec3(enc3(1.467)) isa Number
        # NOTE (TODO §1.4b): enc2/dec2 are built over *different* random ladders,
        # so decode(encode(x)) ≈ x does not hold on the instance path — flagged,
        # not asserted here.
    end

    @testset "FHRR numbers" begin

        v = FHRR()

        numvals = 0:0.1:10

        encoder, decoder = convertlevel(v, numvals)

        x, y, z = 2, 5, 10

        hx, hy, hz = encoder.((x, y, z))

        @test hx isa FHRR
        @test similarity(hx, hy) > similarity(hx, hz)

        @test decoder(hx) < decoder(hy) < decoder(hz)
    end
end
