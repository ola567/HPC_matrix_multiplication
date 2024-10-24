#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void ins__printtime(struct timeval *start, struct timeval *stop) {

  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;
  printf("Execution time = %ld microseconds\n", time);
  return;
}
#endif
