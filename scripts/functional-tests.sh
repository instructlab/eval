#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# This test script is laid out as follows:
# - UTILITIES:  utility functions
# - TESTS:      test functions
# - SETUP:      environment setup steps
# - MAIN:       test execution steps
#
# If you are running locally and calling the script multiple times you may want to run like this:
#
# TEST_DIR=/tmp/foo ./scripts/functional-tests.sh

set -ex

#############
# UTILITIES #
#############

clone_taxonomy(){
    if [ ! -d taxonomy ]; then
        git clone https://github.com/instructlab/taxonomy.git
    fi
}

#########
# TESTS #
#########

test_branch_generator(){
    python3 ${SCRIPTDIR}/test_branch_generator.py --test-dir "${TEST_DIR}"
}

#########
# SETUP #
#########

# shellcheck disable=SC2155
export SCRIPTDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# build a prompt string that includes the time, source file, line number, and function name
export PS4='+$(date +"%Y-%m-%d %T") ${BASH_VERSION}:${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

# Support overriding the test directory for local testing otherwise creates a temporary directory
TEST_DIR=${TEST_DIR:-$(mktemp -d)}

export TEST_DIR
export PACKAGE_NAME='instructlab-eval'


########
# MAIN #
########

pushd $TEST_DIR

clone_taxonomy

test_branch_generator


popd
exit 0
