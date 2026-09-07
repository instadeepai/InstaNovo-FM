from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union


class MMseqs2:
    """
    A Python class wrapper for MMseqs2 clustering and search tool.
    """

    def __init__(  # noqa: CCR001
        self,
        output_dir: Union[str, None] = None,
        fasta_file: Union[str, None] = None,
        sequences: Union[List[str], None] = None,
        command: str = "easy-cluster",
        identity_threshold: float = 0.9,
        is_huge_dataset: bool = False,
        tmp_dir: Union[str, None] = None,
        target_file: Union[str, None] = None,
        output_file: Union[str, None] = None,
        remove_tmp: bool = True,
        remove_output: bool = True,
    ):
        """
        Initializes the MMseqs2 object with the provided parameters.

        Parameters:
        -----------
        output_dir : str or None
            The directory where MMseqs2 results will be saved. If not provided, a temporary
            directory is created.
        fasta_file : str or None
            The path to the input FASTA file with protein sequences, or None if using a list
            of sequences.
        sequences : list or None
            A list of sequences to process. If provided, a temporary FASTA file will be created.
        command : str
            The MMseqs2 command to run. Options are 'easy-cluster', 'easy-linclust', 'easy-search'.
            Defaults to 'easy-cluster'.
        identity_threshold : float
            The sequence identity threshold for clustering. Defaults to 0.9.
        is_huge_dataset : bool
            Whether to use options suitable for huge datasets. Defaults to False.
        tmp_dir : str or None
            Directory for temporary files. If not provided, a temporary directory is created.
        target_file : str or None
            The target database file for 'easy-search' command.
        output_file : str or None
            The output file for 'easy-search' command.
        remove_tmp : bool
            Whether to remove the temporary directory after processing.
            Defaults to True.
        remove_output : bool
            Whether to remove the output directory after processing (if temporary).
            Defaults to True.
        """
        # Set output directory; if not specified, create a temporary directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.temp_output_dir = False
        else:
            self.output_dir = Path(tempfile.mkdtemp())
            self.temp_output_dir = True  # Flag to identify if temp dir was used
            print(f"Temporary output directory created at {self.output_dir}")

        # Set temporary directory
        if tmp_dir:
            self.tmp_dir = Path(tmp_dir)
            self.temp_tmp_dir = False
        else:
            self.tmp_dir = Path(tempfile.mkdtemp())
            self.temp_tmp_dir = True
            print(f"Temporary tmp directory created at {self.tmp_dir}")

        self.fasta_file = Path(fasta_file) if fasta_file else None
        self.sequences = sequences
        self.command = command
        self.identity_threshold = identity_threshold
        self.is_huge_dataset = is_huge_dataset
        self.target_file = Path(target_file) if target_file else None
        self.output_file = Path(output_file) if output_file else None

        self.remove_tmp = remove_tmp
        self.remove_output = remove_output

        self.temp_fasta_file = None  # Placeholder for temp file if sequences are provided

        # Validate input
        if self.fasta_file is None and self.sequences is None:
            raise ValueError("Either a fasta_file or a list of sequences must be provided.")
        if self.fasta_file and not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found at {self.fasta_file}")
        if self.command not in ["easy-cluster", "easy-linclust", "easy-search"]:
            raise ValueError(
                "Invalid command. Options are 'easy-cluster', 'easy-linclust', 'easy-search'."
            )
        if self.command == "easy-search":
            if self.target_file is None or not self.target_file.exists():
                raise ValueError("For 'easy-search', a valid target_file must be specified.")
            if self.output_file is None:
                raise ValueError("For 'easy-search', an output_file must be specified.")

    def _create_temp_fasta(self):
        """
        Creates a temporary FASTA file from the list of sequences provided by the user.
        """
        temp_fasta = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", mode="w")
        with temp_fasta as f:
            for idx, sequence in enumerate(self.sequences, start=1):
                f.write(f">sequence_{idx}\n{sequence}\n")
        self.temp_fasta_file = temp_fasta.name
        print(f"Temporary FASTA file created at {self.temp_fasta_file}")

    def _delete_temp_fasta(self):
        """
        Deletes the temporary FASTA file if it exists.
        """
        if self.temp_fasta_file:
            try:
                os.remove(self.temp_fasta_file)
                print(f"Temporary FASTA file {self.temp_fasta_file} deleted.")
            except OSError as e:
                print(f"Error deleting temporary file {self.temp_fasta_file}: {e}")
            finally:
                self.temp_fasta_file = None

    def _build_command(self):
        """
        Constructs the MMseqs2 command with the provided parameters,
        ensuring directory paths have trailing slashes.
        """
        # If a list of sequences was provided, create a temporary fasta file
        if self.sequences:
            self._create_temp_fasta()
            fasta_input = self.temp_fasta_file
        else:
            fasta_input = str(self.fasta_file)

        # Ensure trailing slashes for directory paths
        output_dir = (
            str(self.output_dir) + "/"
            if not str(self.output_dir).endswith("/")
            else str(self.output_dir)
        )
        tmp_dir = (
            str(self.tmp_dir) + "/" if not str(self.tmp_dir).endswith("/") else str(self.tmp_dir)
        )

        command = ["mmseqs", self.command]

        if self.command in ["easy-cluster", "easy-linclust"]:
            # For clustering commands, set identity threshold immediately after command
            command.extend(
                ["--min-seq-id", str(self.identity_threshold), fasta_input, output_dir, tmp_dir]
            )

            # Handle is_huge_dataset
            if self.is_huge_dataset:
                command.extend(["--split-memory-limit", "0"])

        elif self.command == "easy-search":
            # For search command, need query and target
            target_file = str(self.target_file)
            output_file = str(self.output_file)
            command.extend([fasta_input, target_file, output_file, tmp_dir])

        # # Add verbose flag for debugging
        # command.append("--verbose")

        return command

    def _clean_output(self):
        """
        Removes all files from the output directory if it is temporary.

        Raises:
        -------
        OSError
            If there is an error while removing files.
        """
        if self.temp_output_dir:
            try:
                shutil.rmtree(self.output_dir)
                print(f"Temporary output directory {self.output_dir} deleted.")
            except OSError as e:
                print(f"Error deleting temporary output directory {self.output_dir}: {e}")

    def _clean_tmp_dir(self):
        """
        Removes all files from the tmp directory if it is temporary.

        Raises:
        -------
        OSError
            If there is an error while removing files.
        """
        if self.temp_tmp_dir:
            try:
                shutil.rmtree(self.tmp_dir)
                print(f"Temporary tmp directory {self.tmp_dir} deleted.")
            except OSError as e:
                print(f"Error deleting temporary tmp directory {self.tmp_dir}: {e}")

    def parse_clusters(self) -> Dict[str, int]:
        """
        Parses the MMseqs2 clustering output and returns a dictionary mapping each sequence ID
        to an integer cluster ID.

        Returns:
        --------
        Dict[str, int]
            A dictionary where keys are sequence IDs and values are integer cluster IDs.
        """
        # Locate the cluster file in the output directory
        cluster_files = list(self.output_dir.glob("*_cluster.tsv"))
        if not cluster_files:
            raise FileNotFoundError(f"No cluster file found in {self.output_dir}")
        else:
            cluster_file = cluster_files[0]

        # Add type hint for cluster_annotation
        cluster_annotation: Dict[str, List[str]] = {}
        with open(cluster_file, "r") as f:
            for line in f:
                cluster_id, seq_id = line.strip().split("\t")
                if cluster_id not in cluster_annotation:
                    cluster_annotation[cluster_id] = []
                cluster_annotation[cluster_id].append(seq_id)

        # Map each sequence ID to an integer cluster ID
        cluster_id_map = {cluster_id: i for i, cluster_id in enumerate(cluster_annotation.keys())}
        seq_to_cluster = {
            seq_id: cluster_id_map[cluster_id]
            for cluster_id, seq_ids in cluster_annotation.items()
            for seq_id in seq_ids
        }

        return seq_to_cluster

    def run(self) -> List[Union[int, None]]:  # noqa: CCR001
        """
        Runs the MMseqs2 command and returns the list of assigned cluster IDs
        for each input sequence.

        Returns:
        --------
        List[Union[int, None]]
            A list of integer cluster IDs for each input sequence, or
            None if a sequence was not clustered.
        """
        command = self._build_command()
        print(f"Running MMseqs2 with command: {' '.join(command)}")

        try:
            # Use subprocess.Popen to run the command
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            # Check if the command failed
            if process.returncode != 0:
                print(f"MMseqs2 failed with error:\n{stderr}")
                raise RuntimeError(f"MMseqs2 failed with error:\n{stderr}")

            print(f"MMseqs2 Output:\n{stdout}")

        except subprocess.CalledProcessError as e:
            print(f"MMseqs2 encountered an error: {e.stderr}")
            raise RuntimeError(f"MMseqs2 failed with error: {e.stderr}")

        finally:
            # Clean up the temporary fasta file if created
            self._delete_temp_fasta()

        # Parse clustering results
        seq_to_cluster = self.parse_clusters()

        # Return a list of cluster IDs corresponding to the input sequence order
        if self.sequences:
            cluster_ids = [
                seq_to_cluster.get(f"sequence_{i + 1}") for i in range(len(self.sequences))
            ]
        else:
            if self.fasta_file is None:
                raise ValueError("fasta_file is required if sequences are not provided.")
            cluster_ids = [
                seq_to_cluster.get(seq_id) for seq_id in self.fasta_file.read_text().splitlines()
            ]

        # Clean output and tmp directories if needed
        if self.remove_output:
            self._clean_output()
        if self.remove_tmp:
            self._clean_tmp_dir()

        return cluster_ids

