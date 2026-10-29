import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():

    from beta_stability import init_logger
    import beta_stability.config as config_module
    from argparse import Namespace
    import logging
    import os
    from beta_stability.data import results
    from beta_stability.edge_weights import reviewer

    logger = logging.getLogger("beta_stability")

    return Namespace, config_module, init_logger, logger, os, results, reviewer


@app.cell
def _(logger, os, results, reviewer):
    from beta_stability.run import _degenerate_result
    from beta_stability.data.io import EmptyDataframeException, SingleAnnotationException
    from beta_stability.algorithm_preparation import create_algorithm, get_initial_edges

    def run_once(config):
        """Prepare the algorithm named by the config, run it, and write its graph.

        Returns ``(clustering, node2uuid)``.
        """
        logger.info(f"Starting {config['algorithm_type'].upper()} algorithm")
        try:
            algorithm, common_data = create_algorithm(config)
        except EmptyDataframeException as e:
            return _degenerate_result(config, e, empty=True)
        except SingleAnnotationException as e:
            return _degenerate_result(config, e, empty=False)

        output_path = common_data['output_path']
        os.makedirs(output_path, exist_ok=True)

        init_verifiers = common_data.get('init_verifiers', [common_data['verifier_name']])
        logger.info(f"Initializing graph using verifiers: {init_verifiers}")
        for name in init_verifiers:
            embeddings = common_data['embeddings_dict'][name]
            if hasattr(embeddings, 'k'):
                logger.info(f"K-means found {embeddings.k} clusters")

        initial_edges = get_initial_edges(common_data, config)
        logger.info(f"Starting with {len(initial_edges)} initial edges")
        requested_edges = algorithm.step(initial_edges)

        # An algorithm that presents one review per step declares how many steps it
        # may need (manual and thresholded review); the others keep the configured cap.
        max_outer = max(config['algorithm']['max_outer_iterations'],
                        int(getattr(algorithm, 'required_outer_iterations', 0)))
        iteration = 0
        while not algorithm.is_finished() and iteration < max_outer:
            logger.info(f"Outer iteration {iteration}")
            if requested_edges:
                responses, quit_requested = reviewer.responses(
                    requested_edges, common_data['human_reviewer'])
                if quit_requested:
                    logger.info("Human reviewer requested to quit")
                    break
                requested_edges = algorithm.step(responses)
            else:
                logger.info("No edges for human review, continuing algorithm")
                requested_edges = algorithm.step([])
            iteration += 1

        if iteration >= max_outer:
            logger.warning(f"Reached max outer iterations ({max_outer})")
        else:
            logger.info(f"Algorithm converged after {iteration} outer iterations")

        logger.info("Algorithm execution completed")
        algorithm.show_stats()

        clustering, node2cid, graph = algorithm.get_clustering()
        node2uuid = common_data['node2uuid']
        logger.info(f"Saving results to {output_path}")
        results.write_graph(config, clustering, node2cid, node2uuid, graph, output_path)
        return clustering, node2uuid


    return (run_once,)


@app.cell
def _(Namespace, config_module, init_logger, logger, os, results, run_once):
    init_logger()
    args = Namespace(
        config="configs/final/forestelephants/beta_stability_phc100_mod.yaml", base_config=None, save_dir=None, interactive=False, seed=None, prob_human_correct=None, overrides=None
    )
    config = config_module.apply_run_tag(config_module.load(args.config, args.base_config, args.overrides), seed=args.seed, prob_human_correct=args.prob_human_correct)
    # run_from_config(config, save_dir=args.save_dir)
    save_dir = args.save_dir
    output_base = config['data']['output_path']
    save_dir = save_dir or output_base
    combinations = config_module.field_combinations(config)

    if not combinations:
        clustering, node2uuid = run_once(config)
        mapping, _ = results.uuid_to_cluster_map(clustering, node2uuid)
        results.write_annotations(config, mapping, save_dir)

    else:
        mapping, offset = {}, 0
        for field_combo in combinations:
            logger.info(f"Running for field combination: {field_combo}")
            combo_dir = config_module.combination_dir(output_base, field_combo)
            os.makedirs(combo_dir, exist_ok=True)

            clustering, node2uuid = run_once(
                config_module.for_combination(config, field_combo, combo_dir))
            combo_map, offset = results.uuid_to_cluster_map(clustering, node2uuid, offset)
            mapping.update(combo_map)

            # subsequent runs append to the same log
            if isinstance(config.get('logging'), dict):
                config['logging']['file_mode'] = 'a'

        logger.info(f"Combined {len(combinations)} field combinations into {offset} clusters")
        results.write_annotations(config, mapping, save_dir)
    return (config,)


@app.cell
def _(config):
    config
    return


if __name__ == "__main__":
    app.run()
