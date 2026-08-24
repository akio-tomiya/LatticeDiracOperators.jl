if !isdefined(@__MODULE__, :LDO_TEST_COMM)
    const LDO_TEST_MPI_ENABLED =
        lowercase(get(ENV, "LDO_TEST_MPI", "false")) == "true"

    if LDO_TEST_MPI_ENABLED
        @eval using MPI
        MPI.Initialized() || MPI.Init()
        const LDO_TEST_COMM = MPI.COMM_WORLD
        ldo_test_comm_size() = MPI.Comm_size(LDO_TEST_COMM)
        ldo_test_comm_rank() = MPI.Comm_rank(LDO_TEST_COMM)
        ldo_test_allreduce_sum(value) =
            MPI.Allreduce(value, MPI.SUM, LDO_TEST_COMM)
    else
        const LDO_TEST_COMM = LatticeMatrices.SerialCommunicator()
        ldo_test_comm_size() = 1
        ldo_test_comm_rank() = 0
        ldo_test_allreduce_sum(value) = value
    end
end
