#!/bin/sh

if [ "$1" = "usage" ]
then
  echo "usage : ./submitscript.sh localpdf remotesubmission"
  exit 0
fi

if [ "$#" -ne 2 ]
then
  echo "what?"
  echo "usage : ./submitscript.sh localpdf remotesubmission"
  exit 0
fi

echo "making submission directory..."
echo "mkdir $2" | ssh cs270-as@star.cs.berkeley.edu
echo "copying homework over..."
scp $1 cs270-as@star.cs.berkeley.edu:$2/$2.pdf
echo "submitting $2..."
echo "cd $2;yes | submit $2" | ssh cs270-as@star.cs.berkeley.edu
