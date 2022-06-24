from learning import oracle as ora

def construct_oracle(mode, pddl_problem, problem_info, model_poses, **kwargs):
    if mode in ["oracle", "save"]:
        oracle_class = ora.Oracle
    elif mode == "oraclemodel":
        oracle_class = ora.OracleModel
    elif mode == "oracleexpansion":
        oracle_class = ora.OracleModelExpansion
    elif mode == "complexityV3":
        oracle_class = ora.ComplexityModelV3
    elif mode == "complexityandstructure":
        oracle_class = ora.ComplexityModelV3StructureAware
    elif mode == "model":
        oracle_class = ora.Model
    elif mode == "cachingmodel":
        oracle_class = ora.CachingModel
    elif mode == "multiheadmodel":
        oracle_class = ora.MultiHeadModel
    elif mode == "ploiablation":
        oracle_class = ora.PLOIAblation
    elif mode == "complexitycollector":
        oracle_class = ora.ComplexityDataCollector
    elif mode == "complexitoracle":
        oracle_class = ora.OracleAndComplexityModelExpansion
    elif mode == "oracledagger":
        oracle_class = ora.OracleDAggerModel
    elif mode == "normal":
        return None
    elif mode == "statsablation":
        oracle_class = ora.StatsAblation
    else:
        raise ValueError(f"Unrecognized mode {mode}")
    oracle = oracle_class(
        pddl_problem.domain_pddl,
        pddl_problem.stream_pddl,
        pddl_problem.init,
        pddl_problem.goal,
        model_poses=model_poses,
        **kwargs
    )
    oracle.set_run_attr(problem_info.attr)
    return oracle
