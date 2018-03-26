# create a combination of functions to solve for the inverse of  an assummed
# inversible matrix and use cached data if availaible

## Function to import matrix, setters and getters (to store in cache)

makeCacheMatrix <- function(x = matrix()) {
    inverse <- NULL
    set <- function(y) {
        x <<- y
        inverse <<- NULL
    }
    
    get <- function() x
    setInverse <- function(solver) inverse <<- solver
    getInverse <- function() inverse
    list(set = set, get = get, setInverse = setInverse, getInverse = getInverse)
}


## Function to test for cached inverse and display, or calculate if not cached

cacheSolve <- function(x, ...) {
    inverse <- x$getInverse()
    if(!is.null(inverse)){
        message("getting cached data")
        return(inverse)
    }
    d <- x$get()
    inverse <- solve(d,...)
    x$setInverse(inverse)
    inverse      
       
}
