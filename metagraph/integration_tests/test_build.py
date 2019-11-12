import unittest
import subprocess
from subprocess import PIPE
from tempfile import TemporaryDirectory
import glob
import os


"""Test graph construction"""

METAGRAPH = './metagraph'
TEST_DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../tests/data'

graph_file_extension = {'succinct': '.dbg',
                        'bitmap': '.bitmapdbg',
                        'hash': '.orhashdbg',
                        'hashstr': '.hashstrdbg'}


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()

    def test_simple_all_graphs(self):
        """
        Simple build test
        """

        for representation in ['succinct', 'bitmap', 'hash', 'hashstr']:

            construct_command = '{exe} build --graph {repr} -k 20 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000.fa'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension[representation],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 20', params_str[0])
            self.assertEqual('nodes (k): 591997', params_str[1])
            self.assertEqual('canonical mode: no', params_str[2])

    def test_simple_bloom_graph(self):
        """
        Simple build test
        """

        for representation in ['succinct']:

            construct_command = '{exe} build --graph {repr} -k 20 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000.fa'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension[representation],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 20', params_str[0])
            self.assertEqual('nodes (k): 591997', params_str[1])
            self.assertEqual('canonical mode: no', params_str[2])

            convert_command = '{exe} transform -o {outfile} --initialize-bloom {bloom_param} {input}'.format(
                exe=METAGRAPH,
                outfile=self.tempdir.name + '/graph',
                bloom_param='--bloom-fpp 0.1',
                input=self.tempdir.name + '/graph.dbg'
            )
            res = subprocess.run([convert_command], shell=True)
            self.assertEqual(res.returncode, 0)

            convert_command = '{exe} transform -o {outfile} --initialize-bloom {bloom_param} {input}'.format(
                exe=METAGRAPH,
                outfile=self.tempdir.name + '/graph',
                bloom_param='',
                input=self.tempdir.name + '/graph.dbg'
            )
            res = subprocess.run([convert_command], shell=True)
            self.assertEqual(res.returncode, 0)

    def test_simple_all_graphs_canonical(self):
        """
        Build simple canonical graphs
        """

        # TODO: add 'hashstr' once the canonical mode is implemented for it
        for representation in ['succinct', 'bitmap', 'hash']:  # , 'hashstr']:

            construct_command = '{exe} build \
                    --graph {repr} --canonical -k 20 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000.fa'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension[representation],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 20', params_str[0])
            self.assertEqual('nodes (k): 1159851', params_str[1])
            self.assertEqual('canonical mode: yes', params_str[2])

    def test_build_tiny_k(self):
        for representation in ['succinct', 'bitmap', 'hash', 'hashstr']:
            args = [METAGRAPH, 'build', '--graph', representation,
                    '-k', '2',
                    '-o', self.tempdir.name + '/graph',
                    TEST_DATA_DIR + '/transcripts_1000.fa']
            construct_command = ' '.join(args)

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension[representation],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 2', params_str[0])
            self.assertEqual('nodes (k): 16', params_str[1])
            self.assertEqual('canonical mode: no', params_str[2])

    def test_build_tiny_k_canonical(self):
        # TODO: add 'hashstr' once the canonical mode is implemented for it
        for representation in ['succinct', 'bitmap', 'hash']:  # , 'hashstr']:
            args = [METAGRAPH, 'build', '--graph', representation, '--canonical',
                    '-k', '2',
                    '-o', self.tempdir.name + '/graph',
                    TEST_DATA_DIR + '/transcripts_1000.fa']
            construct_command = ' '.join(args)

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension['succinct'],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 2', params_str[0])
            self.assertEqual('nodes (k): 16', params_str[1])
            self.assertEqual('canonical mode: yes', params_str[2])

    def test_build_from_kmc(self):
        for representation in ['succinct', 'bitmap', 'hash', 'hashstr']:
            construct_command = '{exe} build --graph {repr} -k 11 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000_kmc_counters.kmc_suf'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension[representation],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 11', params_str[0])
            self.assertEqual('nodes (k): 469983', params_str[1])
            self.assertEqual('canonical mode: no', params_str[2])

    def test_build_from_kmc_both(self):
        for representation in ['succinct', 'bitmap', 'hash', 'hashstr']:
            construct_command = '{exe} build --graph {repr} -k 11 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000_kmc_counters_both_strands.kmc_suf'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension[representation],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 11', params_str[0])
            self.assertEqual('nodes (k): 802920', params_str[1])
            self.assertEqual('canonical mode: no', params_str[2])

    def test_build_from_kmc_canonical(self):
        for representation in ['succinct', 'bitmap', 'hash']:  # , 'hashstr']:
            construct_command = '{exe} build \
                    --graph {repr} --canonical -k 11 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000_kmc_counters.kmc_suf'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension['succinct'],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 11', params_str[0])
            self.assertEqual('nodes (k): 802920', params_str[1])
            self.assertEqual('canonical mode: yes', params_str[2])

    def test_build_from_kmc_both_canonical(self):
        for representation in ['succinct', 'bitmap', 'hash']:  # , 'hashstr']:
            construct_command = '{exe} build \
                    --graph {repr} --canonical -k 11 -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000_kmc_counters_both_strands.kmc_suf'
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

            stats_command = '{exe} stats {graph}'.format(
                exe=METAGRAPH,
                graph=self.tempdir.name + '/graph' + graph_file_extension['succinct'],
            )
            res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
            self.assertEqual(res.returncode, 0)
            params_str = res.stdout.decode().split('\n')[2:]
            self.assertEqual('k: 11', params_str[0])
            self.assertEqual('nodes (k): 802920', params_str[1])
            self.assertEqual('canonical mode: yes', params_str[2])

    def test_build_chunks_from_kmc(self):
        representation = 'succinct'

        # Build chunks
        for suffix in ['$', 'A', 'C', 'G', 'T']:
            construct_command = '{exe} build --graph {repr} -k 11 --suffix {suffix} -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000_kmc_counters.kmc_suf',
                suffix=suffix
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

        # Concatenate chunks
        construct_command = '{exe} concatenate --len-suffix 1 --graph {repr} -i {chunk_filebase} -o {outfile}'.format(
            exe=METAGRAPH,
            repr=representation,
            chunk_filebase=self.tempdir.name + '/graph',
            outfile=self.tempdir.name + '/graph_from_chunks',
        )

        res = subprocess.run([construct_command], shell=True)
        self.assertEqual(res.returncode, 0)

        # Check graph
        stats_command = '{exe} stats {graph}'.format(
            exe=METAGRAPH,
            graph=self.tempdir.name + '/graph_from_chunks' + graph_file_extension[representation],
        )
        res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
        self.assertEqual(res.returncode, 0)
        params_str = res.stdout.decode().split('\n')[2:]
        self.assertEqual('k: 11', params_str[0])
        self.assertEqual('nodes (k): 469983', params_str[1])
        self.assertEqual('canonical mode: no', params_str[2])

    def test_build_chunks_from_kmc_canonical(self):
        representation = 'succinct'

        # Build chunks
        for suffix in ['$', 'A', 'C', 'G', 'T']:
            construct_command = '{exe} build --graph {repr} --canonical -k 11 \
                    --suffix {suffix} -o {outfile} {input}'.format(
                exe=METAGRAPH,
                repr=representation,
                outfile=self.tempdir.name + '/graph',
                input=TEST_DATA_DIR + '/transcripts_1000_kmc_counters.kmc_suf',
                suffix=suffix
            )

            res = subprocess.run([construct_command], shell=True)
            self.assertEqual(res.returncode, 0)

        # Concatenate chunks
        construct_command = '{exe} concatenate --len-suffix 1 --graph {repr} -i {chunk_filebase} -o {outfile}'.format(
            exe=METAGRAPH,
            repr=representation,
            chunk_filebase=self.tempdir.name + '/graph',
            outfile=self.tempdir.name + '/graph_from_chunks',
        )

        res = subprocess.run([construct_command], shell=True)
        self.assertEqual(res.returncode, 0)

        # Check graph
        stats_command = '{exe} stats {graph}'.format(
            exe=METAGRAPH,
            graph=self.tempdir.name + '/graph_from_chunks' + graph_file_extension[representation],
        )
        res = subprocess.run(stats_command.split(), stdout=PIPE, stderr=PIPE)
        self.assertEqual(res.returncode, 0)
        params_str = res.stdout.decode().split('\n')[2:]
        self.assertEqual('k: 11', params_str[0])
        self.assertEqual('nodes (k): 802920', params_str[1])
        self.assertEqual('canonical mode: yes', params_str[2])


if __name__ == '__main__':
    unittest.main()
