console.log(Math.tan(1));
console.log(getTanDeg(1));

function getTanDeg(deg) {
  var rad = deg * Math.PI/180;
  return Math.tan(rad);
}
