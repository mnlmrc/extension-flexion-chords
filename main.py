import argparse

CASE_FUNCTIONS = {
    'FORCE:single_trial': force_processing.single_trial,
    'FORCE:average': force_processing.average,
    'EMG:csv2df': emg_processing.csv2df,
    'PLOT:force_in_trial': plotting.plot_force_in_trial,
}

def main(what, **kwargs):
    if what in CASE_FUNCTIONS:
        CASE_FUNCTIONS[what](**kwargs)
    else:
        raise ValueError(f"Unknown case: {what}")

# def main():
#     parser = argparse.ArgumentParser(description="Run specific tasks in the project.")
#     parser.add_argument(
#         "--task",
#         type=str,
#         required=True,
#         help="Task to run: preprocess, analyze, visualize"
#     )
#     args = parser.parse_args()
#
#     # Task dispatcher
#     if args.task == "preprocess":
#         preprocess.run()
#     elif args.task == "analyze":
#         analyze.run()
#     elif args.task == "visualize":
#         visualize.run()
#     else:
#         print(f"Unknown task: {args.task}")
#         print("Available tasks: preprocess, analyze, visualize")

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()

    main()
