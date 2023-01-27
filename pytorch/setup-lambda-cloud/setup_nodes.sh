#!/bin/bash

source ./config.sh

ALL_IP=( $HEAD_IP "${WORKER_IP[@]}" )

echo "List of nodes: "
for IP in ${ALL_IP[*]}; do
    echo $IP
done

echo "Generate ssh keys on the head node ------------------------------"
ssh -i $LAMBDA_CLOUD_KEY ubuntu@$HEAD_IP "ssh-keygen -t rsa -N ''"

echo "Add public key to the all nodes ------------------------------"
for IP in ${ALL_IP[*]}; do
    ssh -i $LAMBDA_CLOUD_KEY ubuntu@$HEAD_IP "cat ~/.ssh/id_rsa.pub" | ssh -i $LAMBDA_CLOUD_KEY ubuntu@$IP "cat >> ~/.ssh/authorized_keys"
done

echo "Set NCCL_IB_DISABLE=1 in .bashrc for all nodes ------------------------------"
for IP in ${ALL_IP[*]}; do
    ssh -i $LAMBDA_CLOUD_KEY ubuntu@$IP "echo export NCCL_IB_DISABLE=1 >> .bashrc"
done

echo "Set NCCL_IB_DISABLE=1 in /etc/environment for all nodes ------------------------------"
for IP in ${ALL_IP[*]}; do
    ssh -i $LAMBDA_CLOUD_KEY ubuntu@$IP "echo NCCL_IB_DISABLE=1 | sudo tee -a /etc/environment"
done

echo "Let the head node ssh into the all nodes at least once so in the future it won't ask about fingerprint ------------------------------"
for IP in ${ALL_IP[*]}; do
    ssh -i $LAMBDA_CLOUD_KEY -t ubuntu@$HEAD_IP "echo exit | xargs ssh ubuntu@$IP"
done

echo "Clone example to all nodes ------------------------------"
for IP in ${ALL_IP[*]}; do
    ssh -i $LAMBDA_CLOUD_KEY ubuntu@$IP "git clone https://github.com/LambdaLabsML/examples.git"
done



